//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3"
	"github.com/GetStream/goaec3/internal/aec3"
	"github.com/GetStream/goaec3/internal/bands"
)

// The C++ reference: the prebuilt NEON libwebrtc in the default lane, the
// scalar (kNone, scalar Ooura) source build of all of aec3 under purego.
const ec3Scalar = purego

func TestEc3ConfigFlatten(t *testing.T) {
	cfg := aec3.DefaultEchoCanceller3Config()
	// Ec3FlattenConfig panics when the Go and C++ leaf counts differ.
	if n := len(Ec3FlattenConfig(&cfg)); n == 0 {
		t.Fatal("no leaves")
	}
}

// ---------------------------------------------------------------------------
// Signals.

// ec3Talker is a speech-like source: a voiced harmonic signal with a
// wandering pitch under a syllable envelope, plus some aspiration noise.
type ec3Talker struct {
	rng      *rand.Rand
	rate     float64
	n        int
	phase    float64
	basePch  float64
	syllable float64
	onFrac   float64 // fraction of time the talker is active
	period   int     // activity period in samples
	offset   int
}

func newEc3Talker(seed int64, rate int, basePitch float64, onFrac float64, periodSec float64) *ec3Talker {
	return &ec3Talker{
		rng: rand.New(rand.NewSource(seed)), rate: float64(rate), basePch: basePitch,
		syllable: 4 + float64(seed%3), onFrac: onFrac, period: int(periodSec * float64(rate)),
		offset: int(seed*7919) % max(1, int(periodSec*float64(rate))),
	}
}

func (s *ec3Talker) next() float64 {
	s.n++
	tm := float64(s.n) / s.rate
	active := 1.0
	if s.onFrac < 1 {
		pos := float64((s.n+s.offset)%s.period) / float64(s.period)
		if pos >= s.onFrac {
			active = 0
		}
	}
	env := math.Max(0, math.Sin(math.Pi*s.syllable*tm))
	env *= env
	pitch := s.basePch * (1 + 0.15*math.Sin(2*math.Pi*0.7*tm) + 0.05*math.Sin(2*math.Pi*3.1*tm))
	s.phase += 2 * math.Pi * pitch / s.rate
	var v float64
	for h := 1; h <= 12; h++ {
		if float64(h)*pitch > s.rate/2-500 {
			break
		}
		v += math.Sin(float64(h)*s.phase) / float64(h)
	}
	return active * (env*0.6*v + 0.02*env*s.rng.NormFloat64())
}

// ec3Room renders the echo: the render signal delayed by delaySamples and
// filtered by a sparse, exponentially decaying random room response.
type ec3Room struct {
	taps []int
	gain []float64
	hist []float64
	pos  int
}

func newEc3Room(seed int64, rate, delaySamples int, gain float64) *ec3Room {
	rng := rand.New(rand.NewSource(seed))
	tail := rate * 30 / 1000 // 30 ms of reverberation
	r := &ec3Room{hist: make([]float64, delaySamples+tail+1)}
	r.taps = append(r.taps, delaySamples)
	r.gain = append(r.gain, gain)
	for i := 0; i < 48; i++ {
		k := 1 + rng.Intn(tail)
		r.taps = append(r.taps, delaySamples+k)
		r.gain = append(r.gain, 0.3*gain*rng.NormFloat64()*math.Exp(-float64(k)/(float64(rate)*0.008)))
	}
	return r
}

func (r *ec3Room) next(x float64) float64 {
	r.pos--
	if r.pos < 0 {
		r.pos = len(r.hist) - 1
	}
	r.hist[r.pos] = x
	var y float64
	n := len(r.hist)
	for i, k := range r.taps {
		j := r.pos + k
		if j >= n {
			j -= n
		}
		y += r.gain[i] * r.hist[j]
	}
	return y
}

// ec3Scene generates render and capture frames in [-1, 1].
type ec3Scene struct {
	rate, numRender, numCapture int
	farEnd                      []*ec3Talker
	nearEnd                     *ec3Talker
	rooms                       [][]*ec3Room // [capture][render]
	noise                       *rand.Rand
	stereoRender                func(frame int) bool
	frame                       int
	saturate                    bool
}

func newEc3Scene(rate, numRender, numCapture int, delayMs int, seed int64) *ec3Scene {
	s := &ec3Scene{rate: rate, numRender: numRender, numCapture: numCapture,
		noise: rand.New(rand.NewSource(seed + 99))}
	for ch := 0; ch < numRender; ch++ {
		s.farEnd = append(s.farEnd, newEc3Talker(seed+int64(ch), rate, 110+40*float64(ch), 1, 1))
	}
	s.nearEnd = newEc3Talker(seed+50, rate, 210, 0.3, 2.3)
	for c := 0; c < numCapture; c++ {
		var rr []*ec3Room
		for r := 0; r < numRender; r++ {
			rr = append(rr, newEc3Room(seed+int64(10*c+r), rate, delayMs*rate/1000+3*c+r, 0.4/float64(numRender)))
		}
		s.rooms = append(s.rooms, rr)
	}
	s.stereoRender = func(int) bool { return true }
	return s
}

// next fills render[ch][rate/100] and capture[ch][rate/100].
func (s *ec3Scene) next(render, capture [][]float32) {
	n := s.rate / 100
	stereo := s.stereoRender(s.frame)
	s.frame++
	x := make([]float64, s.numRender)
	for k := 0; k < n; k++ {
		for ch := range x {
			if ch == 0 || stereo {
				x[ch] = 0.5 * s.farEnd[ch].next()
			} else {
				s.farEnd[ch].next()
				x[ch] = x[0]
			}
			render[ch][k] = float32(x[ch])
		}
		near := 0.3 * s.nearEnd.next()
		for c := 0; c < s.numCapture; c++ {
			var y float64
			for r := 0; r < s.numRender; r++ {
				y += s.rooms[c][r].next(x[r])
			}
			y += near + 0.002*s.noise.NormFloat64()
			if s.saturate && k%97 == 0 && s.frame%50 < 3 {
				y = 0.9999
			}
			capture[c][k] = float32(max(-1, min(1, y)))
		}
	}
}

func ec3Frame(numChannels, n int) [][]float32 {
	f := make([][]float32, numChannels)
	for ch := range f {
		f[ch] = make([]float32, n)
	}
	return f
}

func ec3Flat(f [][]float32, dst []float32) []float32 {
	dst = dst[:0]
	for _, ch := range f {
		dst = append(dst, ch...)
	}
	return dst
}

// ec3Schedule returns the API call pattern: 'R' render, 'C' capture. When
// uneven it has random jitter and, with bursts, a render burst (overrun and
// transfer queue overflow), a capture burst (underrun) and a sweep of render
// bursts.
//
// External-delay scenarios run without bursts: there AlignFromExternalDelay
// applies render_call_counter - capture_call_counter + delay, and a render
// burst longer than the render buffer drives that past the buffer size. The
// C++ RenderDelayBuffer then stores a negative block read index
// (OffsetIndex with offset > size, a DCHECK in debug builds) and
// RenderBuffer::GetBlock(-filter_delay) can index before the start of the
// block vector: undefined behavior with no defined reference output.
func ec3Schedule(numCaptures int, uneven, bursts bool) []byte {
	var s []byte
	rng := rand.New(rand.NewSource(5))
	captures := 0
	renders := 0
	didOverflow, didCaptureBurst, didSweep := false, false, false
	for captures < numCaptures {
		if !uneven {
			s = append(s, 'R', 'C')
			captures++
			continue
		}
		switch {
		case bursts && captures >= 300 && !didOverflow:
			didOverflow = true
			// Render burst beyond the transfer queue capacity.
			for renders < captures+130 {
				s = append(s, 'R')
				renders++
			}
		case bursts && captures >= 700 && !didSweep:
			didSweep = true
			// Render bursts of growing length, each followed by as many
			// captures, so that a render overrun lands right before a
			// capture call (the BufferFlush path).
			for burst := 50; burst <= 80; burst += 3 {
				for i := 0; i < burst; i++ {
					s = append(s, 'R')
					renders++
				}
				for i := 0; i < burst; i++ {
					s = append(s, 'C')
					captures++
				}
			}
		case bursts && captures >= 500 && !didCaptureBurst:
			didCaptureBurst = true
			// Capture burst.
			for i := 0; i < 40; i++ {
				s = append(s, 'C')
				captures++
			}
		default:
			j := rng.Intn(4)
			for i := 0; i < j; i++ {
				s = append(s, 'R')
				renders++
			}
			for i := 0; i < j; i++ {
				s = append(s, 'C')
				captures++
			}
			if j == 0 {
				s = append(s, 'R', 'C')
				renders++
				captures++
			}
		}
	}
	return s
}

// ---------------------------------------------------------------------------
// BlockProcessor with a recording echo remover.

type ec3GoRecorder struct {
	rec Ec3BpRecord
}

func (r *ec3GoRecorder) GetMetrics() aec3.EchoControlMetrics {
	return aec3.EchoControlMetrics{EchoReturnLoss: 1, EchoReturnLossEnhancement: 2}
}

func (r *ec3GoRecorder) ProcessCapture(v aec3.EchoPathVariability, sat bool, d *aec3.DelayEstimate,
	rb *aec3.RenderBuffer, linear, capture *aec3.Block) {
	rec := &r.rec
	rec.Called = true
	rec.GainChange = v.GainChange
	rec.DelayChange = int(v.DelayChange)
	rec.ClockDrift = v.ClockDrift
	rec.Saturation = sat
	rec.HasDelay = d != nil
	if d != nil {
		rec.Delay = d.Delay
		rec.Quality = int(d.Quality)
		rec.BlocksSinceLastChange = d.BlocksSinceLastChange
		rec.BlocksSinceLastUpdate = d.BlocksSinceLastUpdate
	}
	rec.Position = rb.Position()
	rec.RenderActivity = rb.GetRenderActivity()
	b := rb.GetBlock(0)
	i := 0
	for band := 0; band < b.NumBands(); band++ {
		for ch := 0; ch < b.NumChannels(); ch++ {
			copy(rec.Block[i:i+64], b.View(band, ch))
			i += 64
		}
	}
	for band := 0; band < capture.NumBands(); band++ {
		for ch := 0; ch < capture.NumChannels(); ch++ {
			x := capture.View(band, ch)
			for k := range x {
				x[k] *= 0.5
			}
		}
	}
	if linear != nil {
		for ch := 0; ch < linear.NumChannels(); ch++ {
			copy(linear.View(0, ch), capture.View(0, ch))
		}
	}
}

func (r *ec3GoRecorder) UpdateEchoLeakageStatus(bool) {}
func (r *ec3GoRecorder) SetCaptureOutputUsage(bool)   {}

func ec3BlockFlat(b *aec3.Block, dst []float32) []float32 {
	dst = dst[:0]
	for band := 0; band < b.NumBands(); band++ {
		for ch := 0; ch < b.NumChannels(); ch++ {
			dst = append(dst, b.View(band, ch)...)
		}
	}
	return dst
}

func ec3BlockSet(b *aec3.Block, src []float32) {
	i := 0
	for band := 0; band < b.NumBands(); band++ {
		for ch := 0; ch < b.NumChannels(); ch++ {
			copy(b.View(band, ch), src[i:i+64])
			i += 64
		}
	}
}

// ec3SplitStream turns full-band frames into 64-sample split-band blocks
// for the block-level tests (FloatS16 scale).
type ec3SplitStream struct {
	splitter *bands.BandSplitter
	full     [][]float32
	split    [][][]float32
	pending  [][][]float32 // [band][ch] samples not yet emitted
}

func newEc3SplitStream(rate, numChannels int) *ec3SplitStream {
	s := &ec3SplitStream{splitter: bands.NewBandSplitter(rate, numChannels), full: ec3Frame(numChannels, rate/100)}
	nb := s.splitter.NumBands()
	s.split = make([][][]float32, nb)
	s.pending = make([][][]float32, nb)
	for b := range s.split {
		s.split[b] = ec3Frame(numChannels, 160)
		s.pending[b] = make([][]float32, numChannels)
	}
	return s
}

func (s *ec3SplitStream) push(frame [][]float32) {
	for ch := range frame {
		bands.FloatToFloatS16Slice(frame[ch], s.full[ch])
	}
	s.splitter.SplitIntoFrequencyBands(s.full, s.split)
	for b := range s.split {
		for ch := range s.split[b] {
			s.pending[b][ch] = append(s.pending[b][ch], s.split[b][ch]...)
		}
	}
}

func (s *ec3SplitStream) pop(dst []float32) bool {
	if len(s.pending[0][0]) < 64 {
		return false
	}
	i := 0
	for b := range s.pending {
		for ch := range s.pending[b] {
			copy(dst[i:i+64], s.pending[b][ch][:64])
			s.pending[b][ch] = s.pending[b][ch][64:]
			i += 64
		}
	}
	return true
}

type ec3BpCase struct {
	name                string
	rate, nRender, nCap int
	external            bool
	bufferDelayMs       int
	linear              bool
	uneven              bool
}

func TestEc3BlockProcessorParity(t *testing.T) {
	var cases []ec3BpCase
	for _, rate := range []int{16000, 32000, 48000} {
		for _, ch := range [][2]int{{1, 1}, {2, 1}, {2, 2}} {
			for _, uneven := range []bool{false, true} {
				cases = append(cases, ec3BpCase{
					name: fmt.Sprintf("%dHz_r%d_c%d_uneven%v", rate, ch[0], ch[1], uneven),
					rate: rate, nRender: ch[0], nCap: ch[1], uneven: uneven, linear: ch[1] == 2,
				})
			}
		}
		cases = append(cases, ec3BpCase{name: fmt.Sprintf("%dHz_external", rate), rate: rate, nRender: 1,
			nCap: 1, external: true, bufferDelayMs: 100, uneven: true})
		cases = append(cases, ec3BpCase{name: fmt.Sprintf("%dHz_external_nodelay", rate), rate: rate,
			nRender: 2, nCap: 1, external: true})
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) { ec3RunBlockProcessor(t, c) })
	}
}

func ec3RunBlockProcessor(t *testing.T, c ec3BpCase) {
	cfg := aec3.DefaultEchoCanceller3Config()
	cfg.Delay.UseExternalDelayEstimator = c.external
	cfg.Filter.ExportLinearAecOutput = c.linear
	nb := aec3.NumBandsForRate(c.rate)

	cpp := NewEc3BlockProcessor(&cfg, c.rate, c.nRender, c.nCap, ec3Scalar)
	defer cpp.Free()
	rec := &ec3GoRecorder{}
	var rdc *aec3.RenderDelayController
	if !c.external {
		rdc = aec3.NewRenderDelayController(&cfg, c.rate, c.nCap)
	}
	bp := aec3.NewBlockProcessorWithComponents(&cfg, c.rate, c.nRender, c.nCap,
		aec3.NewRenderDelayBuffer(&cfg, c.rate, c.nRender), rdc, rec)
	if c.bufferDelayMs > 0 {
		cpp.SetAudioBufferDelay(c.bufferDelayMs)
		bp.SetAudioBufferDelay(c.bufferDelayMs)
	}

	scene := newEc3Scene(c.rate, c.nRender, c.nCap, 100, 1)
	scene.saturate = true
	renderIn := newEc3SplitStream(c.rate, c.nRender)
	captureIn := newEc3SplitStream(c.rate, c.nCap)
	renderFrame := ec3Frame(c.nRender, c.rate/100)
	captureFrame := ec3Frame(c.nCap, c.rate/100)

	renderBlock := aec3.NewBlock(nb, c.nRender)
	captureBlock := aec3.NewBlock(nb, c.nCap)
	var linearBlock *aec3.Block
	if c.linear {
		b := aec3.NewBlock(1, c.nCap)
		linearBlock = &b
	}
	rbuf := make([]float32, nb*c.nRender*64)
	cbuf := make([]float32, nb*c.nCap*64)
	cppCapture := make([]float32, len(cbuf))
	cppLinear := make([]float32, c.nCap*64)
	var goOut, goLin []float32

	// Each 10 ms frame yields 2.5 blocks on average; the schedule runs in
	// frames, the processors in blocks.
	sched := ec3Schedule(1000, c.uneven, !c.external)
	calls := 0
	sawDelay, sawNewDelay, sawFlush := 0, 0, 0
	var queued [][][]float32
	for frame, op := range sched {
		if op == 'R' {
			scene.next(renderFrame, captureFrame)
			renderIn.push(renderFrame)
			for renderIn.pop(rbuf) {
				ec3BlockSet(&renderBlock, rbuf)
				cpp.BufferRender(rbuf)
				bp.BufferRender(&renderBlock)
			}
			q := ec3Frame(c.nCap, c.rate/100)
			for ch := range q {
				copy(q[ch], captureFrame[ch])
			}
			queued = append(queued, q)
			continue
		}
		if len(queued) > 0 {
			captureIn.push(queued[0])
			queued = queued[1:]
		} else {
			for ch := range captureFrame {
				for k := range captureFrame[ch] {
					captureFrame[ch][k] = float32(0.01 * math.Sin(float64(frame*len(captureFrame[ch])+k)*0.05))
				}
			}
			captureIn.push(captureFrame)
		}
		for captureIn.pop(cbuf) {
			calls++
			gain := calls%173 == 0
			sat := calls%211 < 2
			copy(cppCapture, cbuf)
			want := cpp.ProcessCapture(gain, sat, cppCapture, cppLinear)
			ec3BlockSet(&captureBlock, cbuf)
			rec.rec = Ec3BpRecord{}
			bp.ProcessCapture(gain, sat, linearBlock, &captureBlock)
			got := rec.rec
			if want.HasDelay {
				sawDelay++
			}
			if want.DelayChange == int(aec3.DelayAdjustmentNewDetectedDelay) {
				sawNewDelay++
			}
			if want.DelayChange == int(aec3.DelayAdjustmentBufferFlush) {
				sawFlush++
			}
			if got != want {
				t.Fatalf("block %d: record mismatch\n got %+v\nwant %+v", calls, ec3RecSummary(got), ec3RecSummary(want))
			}
			goOut = ec3BlockFlat(&captureBlock, goOut)
			for i := range goOut {
				if math.Float32bits(goOut[i]) != math.Float32bits(cppCapture[i]) {
					t.Fatalf("block %d: capture[%d] %v want %v", calls, i, goOut[i], cppCapture[i])
				}
			}
			if c.linear {
				goLin = ec3BlockFlat(linearBlock, goLin)
				for i := range goLin {
					if math.Float32bits(goLin[i]) != math.Float32bits(cppLinear[i]) {
						t.Fatalf("block %d: linear[%d] %v want %v", calls, i, goLin[i], cppLinear[i])
					}
				}
			}
			if gm, cm := bp.GetMetrics(), cpp.Metrics(); gm != cm {
				t.Fatalf("block %d: metrics %+v want %+v", calls, gm, cm)
			}
		}
	}
	if calls < 1000 {
		t.Fatalf("only %d capture blocks", calls)
	}
	t.Logf("%d blocks: %d with a delay estimate, %d new delays, %d buffer flushes", calls, sawDelay,
		sawNewDelay, sawFlush)
	if !c.external && (sawDelay == 0 || sawNewDelay == 0) {
		t.Fatal("the delay estimator never produced a delay")
	}

}

func ec3RecSummary(r Ec3BpRecord) string {
	var sum float64
	for _, v := range r.Block {
		sum += float64(v)
	}
	r.Block = [len(r.Block)]float32{}
	return fmt.Sprintf("%+v blocksum=%v", r, sum)
}

// ---------------------------------------------------------------------------
// EchoCanceller3 end to end, through AudioBuffer like AudioProcessingImpl.

type ec3Case struct {
	name                string
	rate, nRender, nCap int
	mutate              func(cfg *aec3.EchoCanceller3Config)
	multichannel        bool
	bufferDelayMs       int
	stereoSwitch        bool // render starts mono and turns stereo
	uneven              bool
	frames              int
	saturate            bool
	usage               bool
}

func ec3Cases(passthrough bool) []ec3Case {
	var cases []ec3Case
	external := func(cfg *aec3.EchoCanceller3Config) { cfg.Delay.UseExternalDelayEstimator = true }
	for _, rate := range []int{16000, 32000, 48000} {
		for _, ch := range [][2]int{{1, 1}, {2, 1}, {1, 2}, {2, 2}} {
			c := ec3Case{
				name: fmt.Sprintf("%dHz_r%d_c%d", rate, ch[0], ch[1]),
				rate: rate, nRender: ch[0], nCap: ch[1], frames: 1000, saturate: true,
			}
			if passthrough {
				c.mutate = external
			}
			cases = append(cases, c)
		}
		mk := func(name string, nr, nc int, mut func(*aec3.EchoCanceller3Config)) ec3Case {
			return ec3Case{name: fmt.Sprintf("%dHz_%s", rate, name), rate: rate, nRender: nr, nCap: nc,
				frames: 1000, mutate: func(cfg *aec3.EchoCanceller3Config) {
					if passthrough {
						external(cfg)
					}
					if mut != nil {
						mut(cfg)
					}
				}}
		}
		c := mk("uneven", 2, 1, nil)
		c.uneven = true
		cases = append(cases, c)
		cases = append(cases, mk("fixed_capture_delay", 1, 2, func(cfg *aec3.EchoCanceller3Config) {
			cfg.Delay.FixedCaptureDelaySamples = 91
		}))
		cases = append(cases, mk("linear_output", 1, 2, func(cfg *aec3.EchoCanceller3Config) {
			cfg.Filter.ExportLinearAecOutput = true
		}))
		cases = append(cases, mk("hpf_echo_reference", 2, 1, func(cfg *aec3.EchoCanceller3Config) {
			cfg.Filter.HighPassFilterEchoReference = true
		}))
		c = mk("stereo_switch", 2, 2, func(cfg *aec3.EchoCanceller3Config) {
			cfg.MultiChannel.StereoDetectionHysteresisSeconds = 0.5
			cfg.MultiChannel.StereoDetectionTimeoutThresholdSeconds = 3
		})
		c.stereoSwitch = true
		c.multichannel = true
		cases = append(cases, c)
		c = mk("stereo_no_detection", 2, 1, func(cfg *aec3.EchoCanceller3Config) {
			cfg.MultiChannel.DetectStereoContent = false
		})
		c.multichannel = true
		cases = append(cases, c)
		if !passthrough {
			c = mk("external_delay", 1, 1, func(cfg *aec3.EchoCanceller3Config) {
				cfg.Delay.UseExternalDelayEstimator = true
			})
			c.bufferDelayMs = 100
			c.uneven = true
			cases = append(cases, c)
			c = mk("capture_output_unused", 1, 1, nil)
			c.usage = true
			cases = append(cases, c)
		}
	}
	return cases
}

// ec3GoCanceller drives aec3.EchoCanceller3 in FloatS16 with the bands
// package doing AudioBuffer's split and merge.
type ec3GoCanceller struct {
	ec        *aec3.EchoCanceller3
	rate      int
	render    *ec3SplitStream
	capture   *ec3SplitStream
	linear    [][]float32
	hasLinear bool
}

func newEc3GoCanceller(cfg, mc *aec3.EchoCanceller3Config, rate, nr, nc int) *ec3GoCanceller {
	g := &ec3GoCanceller{
		ec:   aec3.NewEchoCanceller3(cfg, mc, rate, nr, nc),
		rate: rate, render: newEc3SplitStream(rate, nr), capture: newEc3SplitStream(rate, nc),
	}
	if cfg.Filter.ExportLinearAecOutput {
		g.linear = ec3Frame(nc, 160)
		g.hasLinear = true
	}
	return g
}

func (g *ec3GoCanceller) analyzeRender(fullS16 [][]float32) {
	r := g.render
	for ch := range fullS16 {
		copy(r.full[ch], fullS16[ch])
	}
	r.splitter.SplitIntoFrequencyBands(r.full, r.split)
	g.ec.AnalyzeRender(r.split)
}

func (g *ec3GoCanceller) processCapture(fullS16 [][]float32) {
	c := g.capture
	for ch := range fullS16 {
		copy(c.full[ch], fullS16[ch])
	}
	g.ec.AnalyzeCapture(c.full)
	c.splitter.SplitIntoFrequencyBands(c.full, c.split)
	var lin [][]float32
	if g.hasLinear {
		lin = g.linear
	}
	g.ec.ProcessCapture(c.split, lin, false)
	c.splitter.MergeFrequencyBands(c.split, c.full)
	for ch := range fullS16 {
		copy(fullS16[ch], c.full[ch])
	}
}

func ec3S16(f [][]float32, dst [][]float32) {
	for ch := range f {
		for k, v := range f[ch] {
			dst[ch][k] = v * 32768
		}
	}
}

func ec3RunCanceller(t *testing.T, c ec3Case, withMetrics bool) {
	cfg := aec3.DefaultEchoCanceller3Config()
	if c.mutate != nil {
		c.mutate(&cfg)
	}
	var mc *aec3.EchoCanceller3Config
	if c.multichannel {
		m := aec3.CreateDefaultMultichannelConfig()
		m.Delay = cfg.Delay
		m.Filter.ExportLinearAecOutput = cfg.Filter.ExportLinearAecOutput
		m.Filter.HighPassFilterEchoReference = cfg.Filter.HighPassFilterEchoReference
		m.MultiChannel = cfg.MultiChannel
		mc = &m
	}
	cpp := NewEc3Canceller(&cfg, mc, c.rate, c.nRender, c.nCap, ec3Scalar)
	defer cpp.Free()
	g := newEc3GoCanceller(&cfg, mc, c.rate, c.nRender, c.nCap)
	if c.bufferDelayMs > 0 {
		cpp.SetAudioBufferDelay(c.bufferDelayMs)
		g.ec.SetAudioBufferDelay(c.bufferDelayMs)
	}

	n := c.rate / 100
	scene := newEc3Scene(c.rate, c.nRender, c.nCap, 100, 2)
	scene.saturate = c.saturate
	if c.stereoSwitch {
		scene.stereoRender = func(frame int) bool { return frame >= 300 && frame < 900 }
	}
	render := ec3Frame(c.nRender, n)
	capture := ec3Frame(c.nCap, n)
	renderS16 := ec3Frame(c.nRender, n)
	captureS16 := ec3Frame(c.nCap, n)
	var flat, cppCap []float32
	cppLin := make([]float32, c.nCap*160)
	var linArg []float32
	if cfg.Filter.ExportLinearAecOutput {
		linArg = cppLin
	}
	var queued [][][]float32 // capture frames waiting for their capture call
	frame := 0
	for _, op := range ec3Schedule(c.frames, c.uneven, c.bufferDelayMs == 0) {
		if op == 'R' {
			scene.next(render, capture)
			ec3S16(render, renderS16)
			flat = ec3Flat(renderS16, flat)
			cpp.AnalyzeRender(flat)
			g.analyzeRender(renderS16)
			q := ec3Frame(c.nCap, n)
			ec3S16(capture, q)
			queued = append(queued, q)
			continue
		}
		if len(queued) > 0 {
			for ch := range captureS16 {
				copy(captureS16[ch], queued[0][ch])
			}
			queued = queued[1:]
		} else {
			// Capture without fresh render: a near-end-only frame.
			for ch := range captureS16 {
				for k := range captureS16[ch] {
					captureS16[ch][k] = float32(500 * math.Sin(float64(frame*n+k)*0.05))
				}
			}
		}
		if c.usage && frame%200 == 100 {
			used := frame%400 != 100
			cpp.SetCaptureOutputUsage(used)
			g.ec.SetCaptureOutputUsage(used)
		}
		cppCap = ec3Flat(captureS16, cppCap)
		cpp.ProcessCapture(cppCap, linArg, false)
		g.processCapture(captureS16)
		for ch := range captureS16 {
			for k, v := range captureS16[ch] {
				if w := cppCap[ch*n+k]; math.Float32bits(v) != math.Float32bits(w) {
					t.Fatalf("frame %d ch %d sample %d: %v want %v", frame, ch, k, v, w)
				}
			}
			if linArg != nil {
				for k, v := range g.linear[ch] {
					if w := cppLin[ch*160+k]; math.Float32bits(v) != math.Float32bits(w) {
						t.Fatalf("frame %d linear ch %d sample %d: %v want %v", frame, ch, k, v, w)
					}
				}
			}
		}
		if withMetrics {
			gm, cm := g.ec.GetMetrics(), cpp.Metrics()
			if math.Float64bits(gm.EchoReturnLoss) != math.Float64bits(cm.EchoReturnLoss) ||
				math.Float64bits(gm.EchoReturnLossEnhancement) != math.Float64bits(cm.EchoReturnLossEnhancement) ||
				gm.DelayMs != cm.DelayMs {
				t.Fatalf("frame %d: metrics %+v want %+v", frame, gm, cm)
			}
		}
		frame++
	}
}

// TestEc3CancellerPassthroughParity runs EchoCanceller3 in external-delay
// mode without an audio buffer delay, where the echo remover is bypassed:
// band splitting, the render transfer queue, framing, the fixed capture
// delay, stereo detection and reconfiguration, the high-pass filter and the
// render buffering events all run, and the full-band output must match.
func TestEc3CancellerPassthroughParity(t *testing.T) {
	for _, c := range ec3Cases(true) {
		t.Run(c.name, func(t *testing.T) { ec3RunCanceller(t, c, false) })
	}
}

// TestEc3CancellerParity is the end-to-end comparison with the echo
// remover: output every frame and metrics.
func TestEc3CancellerParity(t *testing.T) {
	for _, c := range ec3Cases(false) {
		t.Run(c.name, func(t *testing.T) { ec3RunCanceller(t, c, true) })
	}
}

// ---------------------------------------------------------------------------
// Public API: [-1, 1] frames through AudioBuffer::CopyFrom / CopyTo.

func ec3RunPublic(t *testing.T, rate, nr, nc int, passthrough, linear bool) {
	cfg := goaec3.DefaultConfig()
	cfg.Delay.UseExternalDelayEstimator = passthrough
	cfg.Filter.ExportLinearAecOutput = linear
	cpp := NewEc3Canceller(&cfg, nil, rate, nr, nc, ec3Scalar)
	defer cpp.Free()
	e, err := goaec3.New(cfg, rate, nr, nc)
	if err != nil {
		t.Fatal(err)
	}
	n := rate / 100
	scene := newEc3Scene(rate, nr, nc, 100, 3)
	scene.saturate = true
	render := ec3Frame(nr, n)
	capture := ec3Frame(nc, n)
	lin := ec3Frame(nc, goaec3.LinearOutputFrameSize)
	cppLin := make([]float32, nc*160)
	var flat, cppCap []float32
	for frame := 0; frame < 1000; frame++ {
		scene.next(render, capture)
		// Out-of-range samples exercise the clamping in the conversion.
		if frame%37 == 0 {
			capture[0][5] = 1.25
			render[0][7] = -1.5
		}
		flat = ec3Flat(render, flat)
		cpp.AnalyzeRenderFloat(flat)
		if err := e.AnalyzeRender(render); err != nil {
			t.Fatal(err)
		}
		cppCap = ec3Flat(capture, cppCap)
		if linear {
			cpp.ProcessCaptureFloat(cppCap, cppLin, false)
			err = e.ProcessCaptureWithLinearOutput(capture, lin)
		} else {
			cpp.ProcessCaptureFloat(cppCap, nil, false)
			err = e.ProcessCapture(capture)
		}
		if err != nil {
			t.Fatal(err)
		}
		for ch := range capture {
			for k, v := range capture[ch] {
				if w := cppCap[ch*n+k]; math.Float32bits(v) != math.Float32bits(w) {
					t.Fatalf("frame %d ch %d sample %d: %v want %v", frame, ch, k, v, w)
				}
			}
			if linear {
				for k, v := range lin[ch] {
					if w := cppLin[ch*160+k]; math.Float32bits(v) != math.Float32bits(w) {
						t.Fatalf("frame %d linear ch %d sample %d: %v want %v", frame, ch, k, v, w)
					}
				}
			}
		}
		if !passthrough {
			gm, cm := e.Metrics(), cpp.Metrics()
			if math.Float64bits(gm.EchoReturnLoss) != math.Float64bits(cm.EchoReturnLoss) ||
				math.Float64bits(gm.EchoReturnLossEnhancement) != math.Float64bits(cm.EchoReturnLossEnhancement) ||
				gm.DelayMs != cm.DelayMs {
				t.Fatalf("frame %d: metrics %+v want %+v", frame, gm, cm)
			}
		}
	}
}

func TestEc3PublicAPIPassthroughParity(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		for _, ch := range [][2]int{{1, 1}, {2, 2}} {
			for _, linear := range []bool{false, true} {
				t.Run(fmt.Sprintf("%dHz_r%d_c%d_linear%v", rate, ch[0], ch[1], linear), func(t *testing.T) {
					ec3RunPublic(t, rate, ch[0], ch[1], true, linear)
				})
			}
		}
	}
}

func TestEc3PublicAPIParity(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		for _, ch := range [][2]int{{1, 1}, {2, 1}, {2, 2}} {
			for _, linear := range []bool{false, true} {
				t.Run(fmt.Sprintf("%dHz_r%d_c%d_linear%v", rate, ch[0], ch[1], linear), func(t *testing.T) {
					ec3RunPublic(t, rate, ch[0], ch[1], false, linear)
				})
			}
		}
	}
}
