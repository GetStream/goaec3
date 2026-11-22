package goaec3

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

// ---------------------------------------------------------------------------
// Signals.

// Signal kinds for one frame. robustSigNaN is the only kind with NaN
// samples; FloatToFloatS16 clamps ±Inf to ±1, so every other kind feeds
// finite values to AEC3.
const (
	robustSigEcho      = iota // render: noise bursts and tones; capture: its echo plus noise
	robustSigSilence          // all zeros
	robustSigFullScale        // ±1 square wave
	robustSigClipping         // the echo scene scaled far beyond ±1
	robustSigDenormal         // subnormal values
	robustSigInf              // the echo scene with ±Inf samples
	robustSigNegZero          // all -0
	robustSigSatEdge          // around the ±32700 saturation threshold
	robustSigDC               // constant 0.5
	robustSigTiny             // noise at 1e-6
	robustSigLoudNoise        // uniform noise in [-1, 1]
	robustSigNaN              // the echo scene with NaN samples
	robustNumSignals
)

// robustScene generates render frames and capture frames that hold the echo
// of the render signal: channel 0 of the render signal, delayed by
// delaySamples and scaled by gain, plus near-end noise.
type robustScene struct {
	rng          *rand.Rand
	rate         int
	n            int
	delaySamples int
	gain         float32
	hist         []float32 // render channel 0 history, newest last
	phase        float64
	t            int
}

func newRobustScene(seed int64, rate, delayMs int) *robustScene {
	d := delayMs * rate / 1000
	return &robustScene{
		rng: rand.New(rand.NewSource(seed)), rate: rate, n: rate / 100, delaySamples: d, gain: 0.5,
		hist: make([]float32, d+rate/100),
	}
}

func (s *robustScene) renderSample() float32 {
	s.t++
	// One second of noise bursts alternating with a swept tone.
	if (s.t/s.rate)%2 == 0 {
		return float32(0.2 * s.rng.NormFloat64())
	}
	s.phase += 2 * math.Pi * (300 + 200*math.Sin(float64(s.t)/float64(s.rate))) / float64(s.rate)
	return float32(0.3 * math.Sin(s.phase))
}

func robustSpecial(rng *rand.Rand, kind int, v float32) float32 {
	switch kind {
	case robustSigSilence:
		return 0
	case robustSigFullScale:
		if v >= 0 {
			return 1
		}
		return -1
	case robustSigClipping:
		return v * 8
	case robustSigDenormal:
		return math.Float32frombits(uint32(rng.Intn(1<<23)) | uint32(rng.Intn(2))<<31)
	case robustSigInf:
		if rng.Intn(16) == 0 {
			return float32(math.Inf(1 - 2*rng.Intn(2)))
		}
		return v
	case robustSigNegZero:
		return float32(math.Copysign(0, -1))
	case robustSigSatEdge:
		e := []float32{32700.0 / 32768, 32699.0 / 32768, 32767.0 / 32768, 1}[rng.Intn(4)]
		if rng.Intn(2) == 0 {
			return -e
		}
		return e
	case robustSigDC:
		return 0.5
	case robustSigTiny:
		return float32(1e-6 * rng.NormFloat64())
	case robustSigLoudNoise:
		return float32(rng.Float64()*2 - 1)
	case robustSigNaN:
		if rng.Intn(16) == 0 {
			return float32(math.NaN())
		}
		return v
	}
	return v
}

// render fills frame[ch][n].
func (s *robustScene) render(frame [][]float32, kind int) {
	copy(s.hist, s.hist[s.n:])
	h := s.hist[len(s.hist)-s.n:]
	for k := 0; k < s.n; k++ {
		x := s.renderSample()
		h[k] = x
		for ch := range frame {
			v := x
			if ch > 0 {
				v = x*0.8 + float32(0.01*s.rng.NormFloat64()*float64(ch))
			}
			frame[ch][k] = robustSpecial(s.rng, kind, v)
		}
	}
}

// capture fills frame[ch][n] with the echo of the last rendered frames.
func (s *robustScene) capture(frame [][]float32, kind int) {
	for k := 0; k < s.n; k++ {
		e := s.gain * s.hist[k]
		for ch := range frame {
			v := e + float32(0.001*s.rng.NormFloat64())
			if ch > 0 {
				v *= 0.9
			}
			frame[ch][k] = robustSpecial(s.rng, kind, v)
		}
	}
}

func robustFrame(numChannels, n int) [][]float32 {
	f := make([][]float32, numChannels)
	for ch := range f {
		f[ch] = make([]float32, n)
	}
	return f
}

func robustHasNaN(frame [][]float32) bool {
	for _, x := range frame {
		for _, v := range x {
			if v != v {
				return true
			}
		}
	}
	return false
}

func robustCheckFinite(t testing.TB, what string, frame [][]float32) {
	t.Helper()
	for ch, x := range frame {
		for k, v := range x {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s: ch %d sample %d is %v", what, ch, k, v)
			}
			if v > 1 || v < -1 {
				t.Fatalf("%s: ch %d sample %d is %v, outside [-1, 1]", what, ch, k, v)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Config leaves.

type robustLeaf struct {
	path string
	v    reflect.Value
}

func robustLeaves(cfg *Config) []robustLeaf {
	var out []robustLeaf
	var walk func(v reflect.Value, path string)
	walk = func(v reflect.Value, path string) {
		switch v.Kind() {
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				walk(v.Field(i), path+"."+v.Type().Field(i).Name)
			}
		case reflect.Bool, reflect.Int, reflect.Float32:
			out = append(out, robustLeaf{path[1:], v})
		default:
			panic(fmt.Sprintf("unsupported config field kind %v at %s", v.Kind(), path))
		}
	}
	walk(reflect.ValueOf(cfg).Elem(), "")
	return out
}

// Caps on the config fields that size buffers and the work per block, for
// the fuzz executions that run many frames. The other executions cap them at
// the limits of New.
const (
	robustMaxFilterBlocks = 40
	robustMaxNumFilters   = 12
)

// robustCapResources lowers the filter lengths to maxBlocks and the number
// of delay filters to maxFilters, and keeps the config valid.
func robustCapResources(cfg *Config, maxBlocks, maxFilters int) {
	f := &cfg.Filter
	f.Refined.LengthBlocks = min(f.Refined.LengthBlocks, maxBlocks)
	f.RefinedInitial.LengthBlocks = min(f.RefinedInitial.LengthBlocks, f.Refined.LengthBlocks)
	f.Coarse.LengthBlocks = min(f.Coarse.LengthBlocks, maxBlocks)
	f.CoarseInitial.LengthBlocks = min(f.CoarseInitial.LengthBlocks, f.Coarse.LengthBlocks)
	cfg.Erle.NumSections = min(cfg.Erle.NumSections, f.Refined.LengthBlocks)
	cfg.Delay.NumFilters = min(cfg.Delay.NumFilters, maxFilters)
}

// robustFrameBudget returns the number of frames a fuzz execution may run
// with cfg: robustMaxFrames up to the cost of the capped configs, and fewer
// in proportion above it.
func robustFrameBudget(cfg *Config, numRender, numCapture int) int {
	const capped = 8 * 8 * robustMaxFilterBlocks
	blocks := min(max(cfg.Filter.Refined.LengthBlocks, cfg.Filter.Coarse.LengthBlocks, 1), MaxFilterLengthBlocks)
	filters := min(max(cfg.Delay.NumFilters, 1), MaxDelayFilters)
	cost := max(numRender*numCapture*blocks, capped*filters/robustMaxNumFilters)
	if cost <= capped {
		return robustMaxFrames
	}
	return max(4, robustMaxFrames*capped/cost)
}

// ---------------------------------------------------------------------------
// Fuzz input decoding.

type robustReader struct {
	b []byte
}

func (r *robustReader) empty() bool { return len(r.b) == 0 }

func (r *robustReader) u8() byte {
	if len(r.b) == 0 {
		return 0
	}
	v := r.b[0]
	r.b = r.b[1:]
	return v
}

func (r *robustReader) u16() uint16 { return uint16(r.u8()) | uint16(r.u8())<<8 }

func (r *robustReader) u32() uint32 {
	var buf [4]byte
	for i := range buf {
		buf[i] = r.u8()
	}
	return binary.LittleEndian.Uint32(buf[:])
}

var robustIntBoundaries = []int{0, 1, 2, 3, 4, 5, 8, 12, 13, 32, 63, 64, 65, 100, 250, 1000, 1024, 5000,
	10000, 100000, 250000, -1, math.MaxInt32, math.MinInt32}

var robustFloatBoundaries = []float32{0, 1, 0.5, 2, 100, 1000, 1e6, 1e8, 1e24, 32768 * 32768, -1, -200,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), math.SmallestNonzeroFloat32,
	math.MaxFloat32}

// robustMutate sets one config leaf from the fuzz input over the full range
// of its type.
func robustMutate(r *robustReader, leaves []robustLeaf) {
	l := leaves[int(r.u16())%len(leaves)]
	sel := r.u8()
	raw := r.u32()
	switch l.v.Kind() {
	case reflect.Bool:
		l.v.SetBool(raw&1 != 0)
	case reflect.Int:
		var x int
		switch sel % 6 {
		case 0:
			x = int(raw % 8)
		case 1:
			x = int(int32(raw))
		case 2:
			x = int(raw % 300)
		case 3:
			x = -1 - int(raw%4)
		case 4:
			x = robustIntBoundaries[int(raw)%len(robustIntBoundaries)]
		default:
			x = int(l.v.Int()) + int(int8(raw))
		}
		l.v.SetInt(int64(x))
	case reflect.Float32:
		var x float32
		switch sel % 7 {
		case 0:
			x = 0
		case 1:
			x = math.Float32frombits(raw)
		case 2:
			x = float32(raw%1000) / 100
		case 3:
			x = -float32(raw%1000) / 100
		case 4:
			x = robustFloatBoundaries[int(raw)%len(robustFloatBoundaries)]
		case 5:
			x = float32(l.v.Float()) * float32(raw%400) / 100
		default:
			x = float32(raw % 1000000)
		}
		l.v.SetFloat(float64(x))
	}
}

// Ops of a fuzzed call sequence.
const (
	robustOpRender = iota
	robustOpCapture
	robustOpPairs
	robustOpRenderBurst
	robustOpCaptureBurst
	robustOpSetDelay
	robustOpOutputUsage
	robustOpGainChange
	robustOpMetrics
	robustOpBadShape   // malformed frames: every call returns an error
	robustOpAliased    // capture in the buffers of the render frame
	robustOpConcurrent // AnalyzeRender on another goroutine during capture
	robustNumOps
)

// Flags of a fuzz input.
const (
	robustFlagMultichannelDefaults = 1 << iota
	robustFlagLinearOutput
	robustFlagExternalDelay
	robustFlagMultichannelConfig
	robustFlagRawConfig // the mutated config goes to New without Validate
	robustFlagLarge     // resources up to the limits of New, fewer frames
)

type robustOp struct {
	kind int
	n    int  // frame count, delay or usage
	sig  int  // signal kind
	lin  bool // capture with linear output
}

type robustProgram struct {
	rate, numRender, numCapture int
	cfg, mc                     Config
	useMC                       bool
	seed                        int64
	delayMs                     int
	ops                         []robustOp
}

func (p *robustProgram) String() string {
	return fmt.Sprintf("rate %d render %d capture %d mc %v ops %d", p.rate, p.numRender, p.numCapture, p.useMC,
		len(p.ops))
}

func robustChannels(b byte) int {
	switch {
	case b < 200:
		return 1 + int(b%2)
	case b < 240:
		return 1 + int(b%8)
	}
	// 9 to MaxChannels+1: New rejects the last.
	return 9 + int(b-240)%9
}

// robustMaxFrames bounds the frames of one fuzz execution.
const robustMaxFrames = 800

// robustDecode turns fuzz input into a program: rate, channel counts, a
// config (mutated and, without robustFlagRawConfig, validated and capped),
// an optional agreeing multichannel config and a call sequence.
func robustDecode(data []byte) robustProgram {
	r := &robustReader{b: data}
	var p robustProgram
	p.rate = []int{16000, 32000, 48000}[int(r.u8())%3]
	p.numRender = robustChannels(r.u8())
	p.numCapture = robustChannels(r.u8())
	flags := r.u8()
	p.seed = int64(r.u8())
	p.delayMs = int(r.u8()) % 200
	p.cfg = DefaultConfig()
	if flags&robustFlagMultichannelDefaults != 0 {
		p.cfg = MultichannelConfig()
	}
	p.cfg.Filter.ExportLinearAecOutput = flags&robustFlagLinearOutput != 0
	p.cfg.Delay.UseExternalDelayEstimator = flags&robustFlagExternalDelay != 0
	p.useMC = flags&robustFlagMultichannelConfig != 0
	raw := flags&robustFlagRawConfig != 0
	maxBlocks, maxFilters := robustMaxFilterBlocks, robustMaxNumFilters
	if flags&robustFlagLarge != 0 {
		// One past the limits: New rejects those.
		maxBlocks, maxFilters = MaxFilterLengthBlocks+1, MaxDelayFilters+1
	}
	leaves := robustLeaves(&p.cfg)
	for range int(r.u8() % 12) {
		robustMutate(r, leaves)
	}
	if !raw {
		p.cfg.Validate()
		robustCapResources(&p.cfg, maxBlocks, maxFilters)
	}
	maxFrames := robustFrameBudget(&p.cfg, p.numRender, p.numCapture)
	if p.useMC {
		p.mc = MultichannelConfig()
		mcLeaves := robustLeaves(&p.mc)
		for range int(r.u8() % 6) {
			robustMutate(r, mcLeaves)
		}
		if !raw {
			p.mc.Validate()
			robustCapResources(&p.mc, maxBlocks, maxFilters)
		}
		robustAgree(&p.mc, &p.cfg)
		maxFrames = min(maxFrames, robustFrameBudget(&p.mc, p.numRender, p.numCapture))
	}

	frames := 0
	for !r.empty() && frames < maxFrames {
		op := robustOp{kind: int(r.u8()) % robustNumOps}
		b := r.u8()
		op.sig = int(b>>1) % robustNumSignals
		op.lin = b&1 != 0
		switch op.kind {
		case robustOpRender, robustOpCapture:
			op.n = 1
		case robustOpPairs:
			op.n = 1 + int(r.u8())*2
		case robustOpRenderBurst, robustOpCaptureBurst:
			op.n = 1 + int(r.u8())%160
		case robustOpConcurrent:
			op.n = 1 + int(r.u8())%40
		case robustOpAliased:
			op.n = 1
		case robustOpSetDelay:
			switch s := r.u8(); s % 4 {
			case 0:
				op.n = int(int32(r.u32()))
			case 1:
				op.n = []int{0, -1, -4, 1, 3, 4, 1000, 10000, math.MaxInt32, math.MinInt32,
					math.MaxInt64, math.MinInt64}[int(s>>2)%12]
			default:
				op.n = int(r.u16() % 600)
			}
		case robustOpOutputUsage:
			op.n = int(b & 1)
		}
		switch op.kind {
		case robustOpRender, robustOpCapture, robustOpRenderBurst, robustOpCaptureBurst, robustOpAliased:
			op.n = min(op.n, maxFrames)
			frames += op.n
		case robustOpPairs, robustOpConcurrent:
			op.n = min(op.n, maxFrames)
			frames += 2 * op.n
		}
		p.ops = append(p.ops, op)
	}
	return p
}

// robustAgree copies the settings NewWithMultichannelConfig requires the
// two configs to agree on (CompatibleConfigs in config_selector.cc: fixed
// capture delay, linear output, high-pass filter, stereo detection) from
// cfg into mc, and the rest of the delay settings with them.
func robustAgree(mc, cfg *Config) {
	mc.Delay = cfg.Delay
	mc.Filter.ExportLinearAecOutput = cfg.Filter.ExportLinearAecOutput
	mc.Filter.HighPassFilterEchoReference = cfg.Filter.HighPassFilterEchoReference
	mc.MultiChannel = cfg.MultiChannel
}

// robustAccepted reports whether New must accept the arguments of p: the
// channel counts within the limit, and each config valid and outside
// robustRejected.
func robustAccepted(p *robustProgram) bool {
	if p.numRender > MaxChannels || p.numCapture > MaxChannels {
		return false
	}
	ok := func(cfg *Config) bool {
		v := *cfg
		return v.Validate() && !robustRejected(cfg)
	}
	return ok(&p.cfg) && (!p.useMC || ok(&p.mc))
}

// robustRun executes a program on the public API and checks that New
// accepts exactly the documented arguments, that the output stays in
// [-1, 1] and finite while no NaN has been fed, and that the metrics stay
// free of NaN under the same condition.
func robustRun(t testing.TB, p *robustProgram) {
	t.Helper()
	var e *EchoCanceller
	var err error
	if p.useMC {
		e, err = NewWithMultichannelConfig(p.cfg, p.mc, p.rate, p.numRender, p.numCapture)
	} else {
		e, err = New(p.cfg, p.rate, p.numRender, p.numCapture)
	}
	if want := robustAccepted(p); (err == nil) != want {
		t.Fatalf("New error = %v, accepted arguments: %v", err, want)
	}
	if err != nil {
		if !errors.Is(err, ErrConfig) && !errors.Is(err, ErrNumChannels) {
			t.Fatalf("New error = %v", err)
		}
		return
	}
	n := e.FrameSize()
	scene := newRobustScene(p.seed, p.rate, p.delayMs)
	render := robustFrame(p.numRender, n)
	capture := robustFrame(p.numCapture, n)
	linear := robustFrame(p.numCapture, LinearOutputFrameSize)
	linearEnabled := p.cfg.Filter.ExportLinearAecOutput
	sawNaN := !robustRenderPowerFinite(&p.cfg) || (p.useMC && !robustRenderPowerFinite(&p.mc))
	doRender := func(sig int) {
		scene.render(render, sig)
		sawNaN = sawNaN || robustHasNaN(render)
		if err := e.AnalyzeRender(render); err != nil {
			t.Fatal(err)
		}
	}
	doCapture := func(sig int, lin bool) {
		scene.capture(capture, sig)
		sawNaN = sawNaN || robustHasNaN(capture)
		if lin && linearEnabled {
			if err := e.ProcessCaptureWithLinearOutput(capture, linear); err != nil {
				t.Fatal(err)
			}
		} else if err := e.ProcessCapture(capture); err != nil {
			t.Fatal(err)
		}
		if !sawNaN {
			robustCheckFinite(t, "capture output", capture)
			if lin && linearEnabled {
				robustCheckFinite(t, "linear output", linear)
			}
			m := e.Metrics()
			if math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement) {
				t.Fatalf("metrics %+v", m)
			}
		}
	}
	var concurrentScene *robustScene
	concurrentRender := robustFrame(p.numRender, n)
	for _, op := range p.ops {
		switch op.kind {
		case robustOpMetrics:
			m := e.Metrics()
			if !sawNaN && (math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement)) {
				t.Fatalf("metrics %+v", m)
			}
		case robustOpBadShape:
			short := robustFrame(p.numCapture, n-1)
			for name, err := range map[string]error{
				"AnalyzeRender(nil)":            e.AnalyzeRender(nil),
				"AnalyzeRender(extra channel)":  e.AnalyzeRender(robustFrame(p.numRender+1, n)),
				"ProcessCapture(nil)":           e.ProcessCapture(nil),
				"ProcessCapture(short)":         e.ProcessCapture(short),
				"ProcessCapture(nil channel)":   e.ProcessCapture(make([][]float32, p.numCapture)),
				"linear output (short capture)": e.ProcessCaptureWithLinearOutput(short, linear),
				"linear output (full rate)":     e.ProcessCaptureWithLinearOutput(capture, robustFrame(p.numCapture, n+1)),
			} {
				if err == nil {
					t.Fatalf("%s: no error", name)
				}
			}
		case robustOpAliased:
			// The render frame is copied, so the capture call may reuse its
			// buffers; with fewer render channels one buffer is several
			// capture channels.
			scene.render(render, op.sig)
			sawNaN = sawNaN || robustHasNaN(render)
			if err := e.AnalyzeRender(render); err != nil {
				t.Fatal(err)
			}
			aliased := make([][]float32, p.numCapture)
			for ch := range aliased {
				aliased[ch] = render[ch%p.numRender]
			}
			if err := e.ProcessCapture(aliased); err != nil {
				t.Fatal(err)
			}
			if !sawNaN {
				robustCheckFinite(t, "aliased capture output", aliased)
			}
		case robustOpConcurrent:
			if concurrentScene == nil {
				concurrentScene = newRobustScene(p.seed+1, p.rate, p.delayMs)
			}
			done := make(chan error, 1)
			go func() {
				for range op.n {
					concurrentScene.render(concurrentRender, robustSigEcho)
					if err := e.AnalyzeRender(concurrentRender); err != nil {
						done <- err
						return
					}
				}
				done <- nil
			}()
			for range op.n {
				doCapture(robustSigEcho, op.lin)
				e.NotifyEchoPathGainChange()
			}
			if err := <-done; err != nil {
				t.Fatal(err)
			}
		case robustOpRender:
			doRender(op.sig)
		case robustOpCapture:
			doCapture(op.sig, op.lin)
		case robustOpPairs:
			for range op.n {
				doRender(robustSigEcho)
				doCapture(robustSigEcho, op.lin)
			}
		case robustOpRenderBurst:
			for range op.n {
				doRender(op.sig)
			}
		case robustOpCaptureBurst:
			for range op.n {
				doCapture(op.sig, op.lin)
			}
		case robustOpSetDelay:
			e.SetAudioBufferDelay(op.n)
		case robustOpOutputUsage:
			e.SetCaptureOutputUsage(op.n != 0)
		case robustOpGainChange:
			e.NotifyEchoPathGainChange()
		}
	}
}

// robustRenderPowerFinite reports whether the render power stays finite.
// RenderLevels.RenderPowerGainDb is not validated; beyond ~200 dB (or NaN)
// the scaled render power overflows to Inf and the spectra turn NaN, in
// libwebrtc as well, so the output has no finiteness guarantee.
func robustRenderPowerFinite(cfg *Config) bool {
	return cfg.RenderLevels.RenderPowerGainDb <= 200
}
