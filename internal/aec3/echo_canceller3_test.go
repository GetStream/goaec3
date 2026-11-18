package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// ec3FakeRemover records the BlockProcessor -> EchoRemover calls.
type ec3FakeRemover struct {
	calls          int
	variability    []EchoPathVariability
	saturation     []bool
	hasDelay       []bool
	delays         []DelayEstimate
	leakage        bool
	outputUsed     bool
	scaleCapture   float32
	renderPosition []int
}

func (f *ec3FakeRemover) GetMetrics() EchoControlMetrics {
	return EchoControlMetrics{EchoReturnLoss: 1, EchoReturnLossEnhancement: 2, DelayMs: 123}
}

func (f *ec3FakeRemover) ProcessCapture(v EchoPathVariability, sat bool, d *DelayEstimate, rb *RenderBuffer,
	linear, capture *Block) {
	f.calls++
	f.variability = append(f.variability, v)
	f.saturation = append(f.saturation, sat)
	f.hasDelay = append(f.hasDelay, d != nil)
	if d != nil {
		f.delays = append(f.delays, *d)
	} else {
		f.delays = append(f.delays, DelayEstimate{})
	}
	f.renderPosition = append(f.renderPosition, rb.Position())
	if f.scaleCapture != 0 {
		for band := 0; band < capture.NumBands(); band++ {
			for ch := 0; ch < capture.NumChannels(); ch++ {
				x := capture.View(band, ch)
				for k := range x {
					x[k] *= f.scaleCapture
				}
			}
		}
	}
	if linear != nil {
		for ch := 0; ch < linear.NumChannels(); ch++ {
			copy(linear.View(0, ch), capture.View(0, ch))
		}
	}
}

func (f *ec3FakeRemover) UpdateEchoLeakageStatus(l bool) { f.leakage = l }
func (f *ec3FakeRemover) SetCaptureOutputUsage(u bool)   { f.outputUsed = u }

func ec3NewFakeBlockProcessor(cfg *EchoCanceller3Config, rate, nr, nc int) (*BlockProcessor, *ec3FakeRemover) {
	f := &ec3FakeRemover{}
	var dc *RenderDelayController
	if !cfg.Delay.UseExternalDelayEstimator {
		dc = NewRenderDelayController(cfg, rate, nc)
	}
	bp := NewBlockProcessorWithComponents(cfg, rate, nr, nc, NewRenderDelayBuffer(cfg, rate, nr), dc, f)
	return bp, f
}

func TestEc3BlockProcessorSkipsCaptureBeforeRender(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	bp, f := ec3NewFakeBlockProcessor(&cfg, 16000, 1, 1)
	capture := NewBlock(1, 1)
	for i := 0; i < 10; i++ {
		bp.ProcessCapture(false, false, nil, &capture)
	}
	if f.calls != 0 {
		t.Fatalf("echo remover called %d times before render started", f.calls)
	}
	render := NewBlock(1, 1)
	bp.BufferRender(&render)
	bp.ProcessCapture(true, true, nil, &capture)
	if f.calls != 1 || !f.variability[0].GainChange || !f.saturation[0] {
		t.Fatalf("calls=%d variability=%+v saturation=%v", f.calls, f.variability, f.saturation)
	}
	bp.UpdateEchoLeakageStatus(true)
	bp.SetCaptureOutputUsage(true)
	if !f.leakage || !f.outputUsed {
		t.Fatal("leakage/output usage not forwarded")
	}
	m := bp.GetMetrics()
	if m.EchoReturnLoss != 1 || m.EchoReturnLossEnhancement != 2 || m.DelayMs != bp.renderBuffer.Delay()*4 {
		t.Fatalf("metrics %+v", m)
	}
}

func TestEc3BlockProcessorRenderOverrunFlushes(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	bp, f := ec3NewFakeBlockProcessor(&cfg, 16000, 1, 1)
	rng := rand.New(rand.NewSource(3))
	render := NewBlock(1, 1)
	capture := NewBlock(1, 1)
	dlyFillBlock(rng, &render, 1000)
	bp.BufferRender(&render)
	bp.ProcessCapture(false, false, nil, &capture)
	sawFlush := false
	for i := 0; i < 400 && !sawFlush; i++ {
		dlyFillBlock(rng, &render, 1000)
		bp.BufferRender(&render)
		if bp.renderEvent == BufferingEventRenderOverrun {
			bp.ProcessCapture(false, false, nil, &capture)
			sawFlush = f.variability[len(f.variability)-1].DelayChange == DelayAdjustmentBufferFlush
			if !sawFlush {
				t.Fatalf("overrun did not flush: %+v", f.variability[len(f.variability)-1])
			}
		}
	}
	if !sawFlush {
		t.Fatal("no render overrun")
	}
}

func TestEc3BlockProcessorExternalDelay(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	cfg.Delay.UseExternalDelayEstimator = true
	bp, f := ec3NewFakeBlockProcessor(&cfg, 32000, 2, 1)
	render := NewBlock(2, 2)
	capture := NewBlock(2, 1)
	for i := 0; i < 20; i++ {
		bp.BufferRender(&render)
		bp.ProcessCapture(false, false, nil, &capture)
	}
	if f.calls != 0 {
		t.Fatalf("echo remover called %d times without an external delay", f.calls)
	}
	bp.SetAudioBufferDelay(40)
	for i := 0; i < 20; i++ {
		bp.BufferRender(&render)
		bp.ProcessCapture(false, false, nil, &capture)
	}
	if f.calls != 20 {
		t.Fatalf("echo remover called %d times, want 20", f.calls)
	}
	for _, h := range f.hasDelay {
		if h {
			t.Fatal("external delay mode passed a delay estimate")
		}
	}
}

// ec3Signals generates a deterministic speech-like FloatS16 signal.
type ec3Signals struct {
	rng   *rand.Rand
	phase float64
	t     int
}

func (s *ec3Signals) next(amp float64) float32 {
	s.t++
	env := 0.5 + 0.5*math.Sin(2*math.Pi*float64(s.t)/4000)
	s.phase += 2 * math.Pi * (200 + 150*math.Sin(float64(s.t)/700)) / 16000
	return float32(amp * env * (math.Sin(s.phase) + 0.3*s.rng.NormFloat64()))
}

func ec3MakeFrame(numBands, numChannels int) [][][]float32 {
	return ec3NewFrame(numBands, numChannels, SplitBandSize)
}

// TestEc3PassthroughFraming runs EchoCanceller3 in external-delay mode
// without an audio buffer delay, where the echo remover is bypassed, and
// checks that the capture output is the input delayed by the framing
// (64 samples) plus the fixed capture delay.
func TestEc3PassthroughFraming(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		for _, fixed := range []int{0, 37} {
			cfg := DefaultEchoCanceller3Config()
			cfg.Delay.UseExternalDelayEstimator = true
			cfg.Delay.FixedCaptureDelaySamples = fixed
			nb := NumBandsForRate(rate)
			ec := NewEchoCanceller3(&cfg, nil, rate, 2, 2)
			sig := &ec3Signals{rng: rand.New(rand.NewSource(1))}
			render := ec3MakeFrame(nb, 2)
			capture := ec3MakeFrame(nb, 2)
			full := [][]float32{make([]float32, rate/100), make([]float32, rate/100)}
			var in, out [][]float32 // per band of channel 1
			in = make([][]float32, nb)
			out = make([][]float32, nb)
			for frame := 0; frame < 50; frame++ {
				for b := range render {
					for ch := range render[b] {
						for k := range render[b][ch] {
							render[b][ch][k] = sig.next(3000)
							capture[b][ch][k] = sig.next(3000)
						}
					}
					in[b] = append(in[b], capture[b][1]...)
				}
				ec.AnalyzeRender(render)
				ec.AnalyzeCapture(full)
				ec.ProcessCapture(capture, nil, false)
				for b := range capture {
					out[b] = append(out[b], capture[b][1]...)
				}
			}
			delay := BlockSize + fixed
			for b := 0; b < nb; b++ {
				for k := delay; k < len(out[b]); k++ {
					if out[b][k] != in[b][k-delay] {
						t.Fatalf("rate %d fixed %d band %d sample %d: got %v want %v", rate, fixed, b, k,
							out[b][k], in[b][k-delay])
					}
				}
			}
		}
	}
}

func TestEc3SaturationDetection(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	ec := NewEchoCanceller3(&cfg, nil, 16000, 1, 2)
	full := [][]float32{make([]float32, 160), make([]float32, 160)}
	ec.AnalyzeCapture(full)
	if ec.saturatedMicrophoneSignal {
		t.Fatal("silence flagged as saturated")
	}
	full[1][17] = -32700
	ec.AnalyzeCapture(full)
	if !ec.saturatedMicrophoneSignal {
		t.Fatal("saturation not detected")
	}
	full[1][17] = 32699.99
	ec.AnalyzeCapture(full)
	if ec.saturatedMicrophoneSignal {
		t.Fatal("below-threshold sample flagged")
	}
}

func TestEc3RenderTransferQueue(t *testing.T) {
	var q ec3RenderTransferQueue
	q.init(3, 1, 1)
	in := ec3NewFrame(1, 1, SplitBandSize)
	out := ec3NewFrame(1, 1, SplitBandSize)
	for i := 0; i < 4; i++ {
		in[0][0][0] = float32(i)
		ok := q.Insert(&in)
		if ok != (i < 3) {
			t.Fatalf("insert %d: %v", i, ok)
		}
	}
	for i := 0; i < 3; i++ {
		if !q.Remove(&out) || out[0][0][0] != float32(i) {
			t.Fatalf("remove %d: %v", i, out[0][0][0])
		}
	}
	if q.Remove(&out) {
		t.Fatal("remove from empty queue")
	}
}

// TestEc3StereoDetectionSwitch checks that proper stereo render content
// reconfigures the canceller to stereo processing and the multichannel
// config.
func TestEc3StereoDetectionSwitch(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	cfg.Delay.UseExternalDelayEstimator = true
	cfg.MultiChannel.StereoDetectionHysteresisSeconds = 0.1
	mc := CreateDefaultMultichannelConfig()
	mc.Delay.UseExternalDelayEstimator = true
	mc.MultiChannel.StereoDetectionHysteresisSeconds = 0.1
	ec := NewEchoCanceller3(&cfg, &mc, 16000, 2, 1)
	render := ec3MakeFrame(1, 2)
	capture := ec3MakeFrame(1, 1)
	full := [][]float32{capture[0][0]}
	rng := rand.New(rand.NewSource(2))
	step := func(stereo bool) {
		for k := range render[0][0] {
			v := float32(rng.NormFloat64() * 1000)
			render[0][0][k] = v
			render[0][1][k] = v
			if stereo {
				render[0][1][k] = float32(rng.NormFloat64() * 1000)
			}
		}
		ec.AnalyzeRender(render)
		ec.AnalyzeCapture(full)
		ec.ProcessCapture(capture, nil, false)
	}
	for i := 0; i < 30; i++ {
		step(false)
	}
	if ec.StereoRenderProcessingActiveForTesting() || ec.GetActiveConfigForTesting().Filter.Coarse.LengthBlocks != 13 {
		t.Fatal("stereo active on mono content")
	}
	for i := 0; i < 30; i++ {
		step(true)
	}
	if !ec.StereoRenderProcessingActiveForTesting() || ec.GetActiveConfigForTesting().Filter.Coarse.LengthBlocks != 11 {
		t.Fatal("stereo not detected")
	}
	if ec.renderBlock.NumChannels() != 2 || ec.numRenderChannelsToAec != 2 {
		t.Fatal("render path not reconfigured to two channels")
	}
}

func TestEc3RenderDownmix(t *testing.T) {
	frame := ec3MakeFrame(2, 3)
	for b := range frame {
		for ch := range frame[b] {
			for k := range frame[b][ch] {
				frame[b][ch][k] = float32(ch+1) + float32(k)
			}
		}
	}
	view := ec3NewSubFrameView(2, 1)
	ec3FillRenderSubFrameView(true, frame, 1, view)
	for b := range view {
		for k, v := range view[b][0] {
			x := float32(80 + k)
			want := ((x + 1) + (x + 2) + (x + 3)) * (float32(1) / 3)
			if v != want {
				t.Fatalf("band %d k %d: %v want %v", b, k, v, want)
			}
		}
	}
	// The first subframe is untouched.
	if frame[0][0][0] != 1 {
		t.Fatal("downmix touched the other subframe")
	}
}

func TestEc3HighPassFilterReference(t *testing.T) {
	// Direct form 1 recursion in float32, written out independently.
	f := NewHighPassFilter(16000, 1)
	c := ec3HighPassFilterCoefficients16kHz[0]
	rng := rand.New(rand.NewSource(4))
	var x1, x2, y1, y2 float32
	for frame := 0; frame < 20; frame++ {
		x := make([]float32, 160)
		for k := range x {
			x[k] = float32(rng.NormFloat64() * 1000)
		}
		want := make([]float32, 160)
		for k, v := range x {
			y := float32(c.B[0]*v) + float32(c.B[1]*x1) + float32(c.B[2]*x2) - float32(c.A[0]*y1) - float32(c.A[1]*y2)
			x2, x1 = x1, v
			y2, y1 = y1, y
			want[k] = y
		}
		f.Process([][]float32{x})
		for k := range x {
			if math.Float32bits(x[k]) != math.Float32bits(want[k]) {
				t.Fatalf("frame %d sample %d: %v want %v", frame, k, x[k], want[k])
			}
		}
	}
	f.ResetChannels(3)
	if f.NumChannels() != 3 {
		t.Fatal("ResetChannels")
	}
	f.ResetChannels(1)
	if f.NumChannels() != 1 {
		t.Fatal("ResetChannels shrink")
	}
}

// TestEc3PassthroughAllocationFree locks the framing/queue/render-buffering
// path (external delay mode without a buffer delay) to zero allocations.
func TestEc3PassthroughAllocationFree(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	cfg.Delay.UseExternalDelayEstimator = true
	cfg.Delay.FixedCaptureDelaySamples = 11
	cfg.Filter.HighPassFilterEchoReference = true
	cfg.Filter.ExportLinearAecOutput = true
	ec := NewEchoCanceller3(&cfg, nil, 48000, 2, 2)
	render := ec3MakeFrame(3, 2)
	capture := ec3MakeFrame(3, 2)
	linear := [][]float32{make([]float32, 160), make([]float32, 160)}
	full := [][]float32{make([]float32, 480), make([]float32, 480)}
	step := func() {
		ec.AnalyzeRender(render)
		ec.AnalyzeCapture(full)
		ec.ProcessCapture(capture, linear, false)
	}
	for i := 0; i < 20; i++ {
		step()
	}
	if a := testing.AllocsPerRun(100, step); a != 0 {
		t.Fatalf("%v allocations per frame", a)
	}
}

// TestEc3AllocationFree locks the full steady-state path to zero
// allocations.
func TestEc3AllocationFree(t *testing.T) {
	for _, rate := range []int{16000, 48000} {
		cfg := DefaultEchoCanceller3Config()
		cfg.Filter.ExportLinearAecOutput = true
		nb := NumBandsForRate(rate)
		ec := NewEchoCanceller3(&cfg, nil, rate, 2, 2)
		sig := &ec3Signals{rng: rand.New(rand.NewSource(1))}
		render := ec3MakeFrame(nb, 2)
		capture := ec3MakeFrame(nb, 2)
		linear := [][]float32{make([]float32, 160), make([]float32, 160)}
		full := [][]float32{make([]float32, rate/100), make([]float32, rate/100)}
		step := func() {
			for b := range render {
				for ch := range render[b] {
					for k := range render[b][ch] {
						render[b][ch][k] = sig.next(3000)
						capture[b][ch][k] = 0.5 * render[b][ch][k]
					}
				}
			}
			ec.AnalyzeRender(render)
			ec.AnalyzeCapture(full)
			ec.ProcessCapture(capture, linear, false)
			_ = ec.GetMetrics()
		}
		for i := 0; i < 300; i++ {
			step()
		}
		if a := testing.AllocsPerRun(100, step); a != 0 {
			t.Fatalf("rate %d: %v allocations per frame", rate, a)
		}
	}
}
