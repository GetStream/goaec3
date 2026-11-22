//go:build webrtc_oracle || webrtc_src_oracle

package webrtcoracle

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"reflect"
	"strconv"
	"testing"

	"github.com/GetStream/goaec3"
	"github.com/GetStream/goaec3/internal/aec3"
)

// FuzzEc3Differential drives goaec3 and libwebrtc's EchoCanceller3 (through
// AudioBuffer, as AudioProcessingImpl does) with the same fuzzed config,
// multichannel config, rate, channel counts, call sequence and audio, and
// requires bit-exact capture output, linear output and metrics after every
// capture frame.
//
// The generator stays within defined C++ behavior; the exclusions are listed
// at fzDefined, fzExternalDelay and fzDecode.
func FuzzEc3Differential(f *testing.F) {
	for _, s := range fzSeeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		p, ok := fzDecode(data)
		if !ok {
			t.Skip("outside defined C++ behavior")
		}
		fzRun(t, &p)
	})
}

// fzSeeds are the seed programs: a header (rate, render channels, capture
// channels, flags, seed, delay, mutation count[, multichannel mutation
// count]) followed by ops.
var fzSeeds = [][]byte{
	{0, 0, 0, 0, 1, 40, 0, fzOpPairs, 0, 199},
	{2, 1, 0, 2 | 8, 2, 60, 0, 0, fzOpPairs, 1, 150, fzOpRenderBurst, 0, 120,
		fzOpCaptureBurst, 2 * fzSigFullScale, 30, fzOpPairs, 1, 100},
	{1, 0, 0, 4, 3, 80, 0, fzOpSetDelay, 0, 2, 80, 0, fzOpPairs, 0, 60, fzOpRenderBurst, 0, 4,
		fzOpSetDelay, 0, 1, fzOpCaptureBurst, 0, 5, fzOpPairs, 0, 40},
	{0, 0, 0, 0, 4, 20, 0, fzOpPairs, 0, 100,
		fzOpRender, 2 * fzSigInf, fzOpCapture, 2 * fzSigClipping,
		fzOpRender, 2 * fzSigDenormal, fzOpCapture, 2 * fzSigSatEdge,
		fzOpRender, 2 * fzSigNegZero, fzOpCapture, 2 * fzSigDC,
		fzOpGainChange, 0, fzOpOutputUsage, 0, fzOpPairs, 0, 20, fzOpOutputUsage, 1, fzOpPairs, 0, 20},
	{2, 207, 207, 0, 5, 10, 0, fzOpPairs, 0, 60},
	{1, 1, 1, 1 | 2, 6, 100, 0, fzOpPairs, 1, 250, fzOpRenderBurst, 4, 150, fzOpPairs, 1, 50},
}

// Signals.

const (
	fzSigEcho = iota
	fzSigSilence
	fzSigFullScale
	fzSigClipping
	fzSigDenormal
	fzSigInf
	fzSigNegZero
	fzSigSatEdge
	fzSigDC
	fzSigTiny
	fzSigLoudNoise
	fzNumSignals
)

// fzScene generates render noise bursts and a swept tone; its capture holds
// the echo of render channel 0 (delayed and scaled) plus noise.
type fzScene struct {
	rng   *rand.Rand
	rate  int
	n     int
	gain  float32
	hist  []float32
	phase float64
	t     int
}

func newFzScene(seed int64, rate, delayMs int) *fzScene {
	d := delayMs * rate / 1000
	return &fzScene{rng: rand.New(rand.NewSource(seed)), rate: rate, n: rate / 100, gain: 0.5,
		hist: make([]float32, d+rate/100)}
}

func (s *fzScene) renderSample() float32 {
	s.t++
	if (s.t/s.rate)%2 == 0 {
		return float32(0.2 * s.rng.NormFloat64())
	}
	s.phase += 2 * math.Pi * (300 + 200*math.Sin(float64(s.t)/float64(s.rate))) / float64(s.rate)
	return float32(0.3 * math.Sin(s.phase))
}

func fzSpecial(rng *rand.Rand, kind int, v float32) float32 {
	switch kind {
	case fzSigSilence:
		return 0
	case fzSigFullScale:
		if v >= 0 {
			return 1
		}
		return -1
	case fzSigClipping:
		return v * 8
	case fzSigDenormal:
		return math.Float32frombits(uint32(rng.Intn(1<<23)) | uint32(rng.Intn(2))<<31)
	case fzSigInf:
		if rng.Intn(16) == 0 {
			return float32(math.Inf(1 - 2*rng.Intn(2)))
		}
		return v
	case fzSigNegZero:
		return float32(math.Copysign(0, -1))
	case fzSigSatEdge:
		e := []float32{32700.0 / 32768, 32699.0 / 32768, 32767.0 / 32768, 1}[rng.Intn(4)]
		if rng.Intn(2) == 0 {
			return -e
		}
		return e
	case fzSigDC:
		return 0.5
	case fzSigTiny:
		return float32(1e-6 * rng.NormFloat64())
	case fzSigLoudNoise:
		return float32(rng.Float64()*2 - 1)
	}
	return v
}

func (s *fzScene) render(frame [][]float32, kind int) {
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
			frame[ch][k] = fzSpecial(s.rng, kind, v)
		}
	}
}

func (s *fzScene) capture(frame [][]float32, kind int) {
	for k := 0; k < s.n; k++ {
		e := s.gain * s.hist[k]
		for ch := range frame {
			v := e + float32(0.001*s.rng.NormFloat64())
			if ch > 0 {
				v *= 0.9
			}
			frame[ch][k] = fzSpecial(s.rng, kind, v)
		}
	}
}

// fzAudio produces the render frames and the capture frames.
type fzAudio interface {
	render(frame [][]float32, kind int)
	capture(frame [][]float32, kind int)
}

// fzRoomAudio adapts ec3Scene (speech-like far- and near-end talkers,
// sparse random room responses per render/capture channel pair) to fzAudio:
// each render frame generates the matching capture frame, which the next
// capture call consumes.
type fzRoomAudio struct {
	scene   *ec3Scene
	rng     *rand.Rand
	pending [][][]float32
	spare   [][][]float32
	n       int
}

func newFzRoomAudio(p *fzProgram) *fzRoomAudio {
	return &fzRoomAudio{scene: newEc3Scene(p.rate, p.numRender, p.numCapture, p.delayMs, p.seed),
		rng: rand.New(rand.NewSource(p.seed + 1)), n: p.rate / 100}
}

func (a *fzRoomAudio) render(frame [][]float32, kind int) {
	var c [][]float32
	if k := len(a.spare); k > 0 {
		c, a.spare = a.spare[k-1], a.spare[:k-1]
	} else {
		c = ec3Frame(len(a.scene.rooms), a.n)
	}
	a.scene.next(frame, c)
	a.pending = append(a.pending, c)
	for _, x := range frame {
		for k := range x {
			x[k] = fzSpecial(a.rng, kind, x[k])
		}
	}
}

func (a *fzRoomAudio) capture(frame [][]float32, kind int) {
	if len(a.pending) == 0 {
		for _, x := range frame {
			for k := range x {
				x[k] = fzSpecial(a.rng, kind, float32(0.002*a.rng.NormFloat64()))
			}
		}
		return
	}
	c := a.pending[0]
	a.pending = a.pending[1:]
	for ch, x := range frame {
		for k := range x {
			x[k] = fzSpecial(a.rng, kind, c[ch][k])
		}
	}
	a.spare = append(a.spare, c)
}

// Config.

type fzLeaf struct {
	path string
	v    reflect.Value
}

func fzLeaves(cfg *aec3.EchoCanceller3Config) []fzLeaf {
	var out []fzLeaf
	var walk func(v reflect.Value, path string)
	walk = func(v reflect.Value, path string) {
		switch v.Kind() {
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				walk(v.Field(i), path+"."+v.Type().Field(i).Name)
			}
		default:
			out = append(out, fzLeaf{path[1:], v})
		}
	}
	walk(reflect.ValueOf(cfg).Elem(), "")
	return out
}

type fzReader struct{ b []byte }

func (r *fzReader) empty() bool { return len(r.b) == 0 }

func (r *fzReader) u8() byte {
	if len(r.b) == 0 {
		return 0
	}
	v := r.b[0]
	r.b = r.b[1:]
	return v
}

func (r *fzReader) u16() uint16 { return uint16(r.u8()) | uint16(r.u8())<<8 }

func (r *fzReader) u32() uint32 {
	var buf [4]byte
	for i := range buf {
		buf[i] = r.u8()
	}
	return binary.LittleEndian.Uint32(buf[:])
}

var fzIntBoundaries = []int{0, 1, 2, 3, 4, 5, 8, 12, 13, 32, 63, 64, 65, 100, 250, 1000, 1024, 5000,
	10000, 100000, 250000, -1, math.MaxInt32, math.MinInt32}

var fzFloatBoundaries = []float32{0, 1, 0.5, 2, 100, 1000, 1e6, 1e8, 1e24, 32768 * 32768, -1, -200,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), math.SmallestNonzeroFloat32,
	math.MaxFloat32}

func fzMutate(r *fzReader, leaves []fzLeaf) {
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
			x = fzIntBoundaries[int(raw)%len(fzIntBoundaries)]
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
			x = fzFloatBoundaries[int(raw)%len(fzFloatBoundaries)]
		case 5:
			x = float32(l.v.Float()) * float32(raw%400) / 100
		default:
			x = float32(raw % 1000000)
		}
		l.v.SetFloat(float64(x))
	}
}

// fzCapResources bounds the memory and time of an execution; it excludes no
// behavior.
func fzCapResources(cfg *aec3.EchoCanceller3Config) {
	f := &cfg.Filter
	f.Refined.LengthBlocks = min(f.Refined.LengthBlocks, 40)
	f.RefinedInitial.LengthBlocks = min(f.RefinedInitial.LengthBlocks, f.Refined.LengthBlocks)
	f.Coarse.LengthBlocks = min(f.Coarse.LengthBlocks, 40)
	f.CoarseInitial.LengthBlocks = min(f.CoarseInitial.LengthBlocks, f.Coarse.LengthBlocks)
	cfg.Erle.NumSections = min(cfg.Erle.NumSections, f.Refined.LengthBlocks)
	cfg.Delay.NumFilters = min(cfg.Delay.NumFilters, 12)
}

// fzRenderDelayBufferSize is the size of the RenderDelayBuffer block ring.
func fzRenderDelayBufferSize(cfg *aec3.EchoCanceller3Config) int {
	return aec3.GetRenderDelayBufferSize(cfg.Delay.DownSamplingFactor, cfg.Delay.NumFilters,
		cfg.Filter.Refined.LengthBlocks)
}

// fzDefined reports whether the C++ behavior for a config that passed
// Validate is defined. Validate accepts each excluded value; the C++ then
// crashes or reads out of bounds, so there is no reference output:
//   - Delay.NumFilters == 0: the EchoCanceller3 constructor segfaults.
//   - Filter.Refined.LengthBlocks < 9: ReverbDecayEstimator constructs
//     EarlyReverbLengthEstimator(length_blocks - 3), which sizes a vector
//     with max_blocks - 6 < 0 (reverb_decay_estimator.cc), and aborts.
//   - SubbandNearendDetection.Subband{1,2}.High == 65 with
//     UseSubbandNearendDetection: SubbandNearendDetector::Update
//     accumulates up to begin() + high + 1 of a 65-element array.
//   - Delay.DelayHeadroomSamples / 64 >= Filter.RefinedInitial.LengthBlocks:
//     while the delay estimator may not have converged, AecState::FilterDelay
//     reports the headroom in blocks as the filter delay, and
//     ReverbFrequencyResponse::Update reads frequency_response[delay] of a
//     vector that has one element per filter block.
//   - Delay.DefaultDelay >= the render buffer size: RenderDelayBuffer::Reset
//     applies OffsetIndex(write, -default_delay), a negative ring index.
//
// New also rejects a defined config whose output turns NaN on finite input:
//   - ErrorFloor or NoiseGate of Filter.Refined or Filter.RefinedInitial
//     below 1e-12: the step size of the refined filter is 0/0 or infinite
//     once the capture signal is digital silence
//     (refined_filter_update_gain.cc); TestEc3RefinedStepNaN has the C++ do
//     it.
//
// A further exclusion is a value the C++ config cannot hold:
//   - Buffering.ExcessRenderDetectionIntervalBlocks and
//     Buffering.MaxAllowedExcessRenderBlocks < 0: both are size_t in the
//     C++ (Validate clamps every other size_t field to >= 0), so a negative
//     value has no C++ counterpart (the oracle's double -> size_t
//     conversion of it is undefined).
func fzDefined(cfg *aec3.EchoCanceller3Config) bool {
	if cfg.Buffering.ExcessRenderDetectionIntervalBlocks < 0 || cfg.Buffering.MaxAllowedExcessRenderBlocks < 0 {
		return false
	}
	if cfg.Delay.NumFilters == 0 || cfg.Filter.Refined.LengthBlocks < 9 {
		return false
	}
	if cfg.Delay.DelayHeadroomSamples/aec3.BlockSize >= cfg.Filter.RefinedInitial.LengthBlocks {
		return false
	}
	for _, r := range []*aec3.RefinedConfiguration{&cfg.Filter.Refined, &cfg.Filter.RefinedInitial} {
		if r.ErrorFloor < 1e-12 || r.NoiseGate < 1e-12 {
			return false
		}
	}
	sn := &cfg.Suppressor.SubbandNearendDetection
	if cfg.Suppressor.UseSubbandNearendDetection &&
		(sn.Subband1.High >= aec3.FftLengthBy2Plus1 || sn.Subband2.High >= aec3.FftLengthBy2Plus1) {
		return false
	}
	return cfg.Delay.DefaultDelay < fzRenderDelayBufferSize(cfg)
}

// fzExternalDelay tracks the RenderDelayBuffer block counters in external
// delay mode. AlignFromExternalDelay applies the total delay
// render_call_counter - capture_call_counter + external_delay - headroom
// through OffsetIndex(write, -delay) and OffsetIndex(write, delay); once its
// magnitude exceeds the ring size the C++ stores a negative ring index and
// reads out of bounds (render_delay_buffer.cc), so the run stops before a
// capture frame could get there. Render frames reach the counters when the
// capture side empties the transfer queue, which drops frames beyond its
// capacity; capture frames count whether or not render has started. A
// 10 ms frame is 2.5 blocks; the margin covers the block framing.
type fzExternalDelay struct {
	size, headroomBlocks int
	pending              int // frames in the render transfer queue
	renderFrames         int
	captureFrames        int
	hasDelay             bool
	delayBlocks          int
}

func (x *fzExternalDelay) render() {
	if x.pending < aec3.RenderTransferQueueSizeFrames {
		x.pending++
	}
}

func (x *fzExternalDelay) setDelay(delayMs int) {
	x.hasDelay = true
	x.delayBlocks = int(int32(delayMs) / 4)
}

// captureInRange reports whether the next capture frame keeps the total
// delay inside the ring.
func (x *fzExternalDelay) captureInRange() bool {
	if !x.hasDelay {
		return true
	}
	const margin = 4
	r := (x.renderFrames + x.pending) * 5 / 2
	for _, c := range []int{x.captureFrames * 5 / 2, (x.captureFrames + 1) * 5 / 2} {
		d := r - c + x.delayBlocks - x.headroomBlocks
		if d > x.size-margin || d < -(x.size-margin) {
			return false
		}
	}
	return true
}

func (x *fzExternalDelay) capture() {
	x.renderFrames += x.pending
	x.pending = 0
	x.captureFrames++
}

// Program.

const (
	fzOpRender = iota
	fzOpCapture
	fzOpPairs
	fzOpRenderBurst
	fzOpCaptureBurst
	fzOpSetDelay
	fzOpOutputUsage
	fzOpGainChange
	fzNumOps
)

type fzOp struct {
	kind int
	n    int
	sig  int
	lin  bool
}

type fzProgram struct {
	rate, numRender, numCapture int
	cfg, mc                     aec3.EchoCanceller3Config
	useMC                       bool
	seed                        int64
	delayMs                     int
	room                        bool // speech-like talkers through random rooms (ec3Scene)
	ops                         []fzOp
}

func (p *fzProgram) String() string {
	return fmt.Sprintf("rate %d render %d capture %d mc %v ops %d", p.rate, p.numRender, p.numCapture, p.useMC,
		len(p.ops))
}

func fzChannels(b byte) int {
	if b < 200 {
		return 1 + int(b%2)
	}
	return 1 + int(b%8)
}

const fzMaxFrames = 800

// fzAgree copies from cfg into mc the fields the C++ CompatibleConfigs
// (config_selector.cc) requires to match: fixed capture delay, linear
// output, high-pass filter and stereo detection. It copies the whole delay
// config, which keeps one render buffer geometry for fzExternalDelay.
func fzAgree(mc, cfg *aec3.EchoCanceller3Config) {
	mc.Delay = cfg.Delay
	mc.Filter.ExportLinearAecOutput = cfg.Filter.ExportLinearAecOutput
	mc.Filter.HighPassFilterEchoReference = cfg.Filter.HighPassFilterEchoReference
	mc.MultiChannel = cfg.MultiChannel
}

// fzDecode decodes a program. It reports false when the program would leave
// defined C++ behavior.
func fzDecode(data []byte) (fzProgram, bool) {
	r := &fzReader{b: data}
	var p fzProgram
	p.rate = []int{16000, 32000, 48000}[int(r.u8())%3]
	p.numRender = fzChannels(r.u8())
	p.numCapture = fzChannels(r.u8())
	flags := r.u8()
	p.seed = int64(r.u8())
	p.delayMs = int(r.u8()) % 200
	p.cfg = aec3.DefaultEchoCanceller3Config()
	if flags&1 != 0 {
		p.cfg = aec3.CreateDefaultMultichannelConfig()
	}
	p.cfg.Filter.ExportLinearAecOutput = flags&2 != 0
	p.cfg.Delay.UseExternalDelayEstimator = flags&4 != 0
	p.useMC = flags&8 != 0
	leaves := fzLeaves(&p.cfg)
	for range int(r.u8() % 12) {
		fzMutate(r, leaves)
	}
	p.cfg.Validate()
	fzCapResources(&p.cfg)
	if p.cfg.Delay.UseExternalDelayEstimator && p.numRender > 1 {
		// A detected switch to multichannel content re-creates the block
		// processor and with it the render buffer counters that
		// fzExternalDelay tracks; with detection off the content mode is
		// fixed from the start.
		p.cfg.MultiChannel.DetectStereoContent = false
	}
	if p.useMC {
		p.mc = aec3.CreateDefaultMultichannelConfig()
		mcLeaves := fzLeaves(&p.mc)
		for range int(r.u8() % 6) {
			fzMutate(r, mcLeaves)
		}
		p.mc.Validate()
		fzCapResources(&p.mc)
		fzAgree(&p.mc, &p.cfg)
	}
	if !fzDefined(&p.cfg) || (p.useMC && !fzDefined(&p.mc)) {
		return p, false
	}

	frames := 0
	for !r.empty() && frames < fzMaxFrames {
		op := fzOp{kind: int(r.u8()) % fzNumOps}
		b := r.u8()
		op.sig = int(b>>1) % fzNumSignals
		op.lin = b&1 != 0
		switch op.kind {
		case fzOpRender, fzOpCapture:
			op.n = 1
		case fzOpPairs:
			op.n = 1 + int(r.u8())*2
		case fzOpRenderBurst, fzOpCaptureBurst:
			op.n = 1 + int(r.u8())%160
		case fzOpSetDelay:
			switch s := r.u8(); s % 4 {
			case 0:
				op.n = int(int32(r.u32()))
			case 1:
				op.n = []int{0, -1, -4, 1, 3, 4, 1000, 10000, math.MaxInt32, math.MinInt32,
					math.MaxInt64, math.MinInt64}[int(s>>2)%12]
			default:
				op.n = int(r.u16() % 600)
			}
		case fzOpOutputUsage:
			op.n = int(b & 1)
		}
		switch op.kind {
		case fzOpRender, fzOpCapture, fzOpRenderBurst, fzOpCaptureBurst:
			frames += op.n
		case fzOpPairs:
			frames += 2 * op.n
		}
		p.ops = append(p.ops, op)
	}
	return p, true
}

// fzConfigDiff lists the config leaves that differ from def.
func fzConfigDiff(cfg, def *aec3.EchoCanceller3Config) string {
	var b []byte
	dl, cl := fzLeaves(def), fzLeaves(cfg)
	for i := range dl {
		if x, y := fmt.Sprint(dl[i].v.Interface()), fmt.Sprint(cl[i].v.Interface()); x != y {
			b = fmt.Appendf(b, "\n  %s: %s -> %s", cl[i].path, x, y)
		}
	}
	return string(b)
}

// fzSame32 compares bit-exactly, except that any two NaNs are equal: the
// sign and payload of a NaN that an arithmetic chain produces is not part
// of IEEE 754 results, and the port and libwebrtc produce different NaN
// bit patterns once a NaN enters (see TestEc3NaNInputParity).
func fzSame32(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

func fzSame64(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) }

// fzRun executes the program on both implementations and returns the
// number of capture frames processed (fewer than the program has when the
// external delay tracker stops the run).
func fzRun(t *testing.T, p *fzProgram) int {
	var mcPtr *aec3.EchoCanceller3Config
	var e *goaec3.EchoCanceller
	var err error
	if p.useMC {
		mcPtr = &p.mc
		e, err = goaec3.NewWithMultichannelConfig(p.cfg, p.mc, p.rate, p.numRender, p.numCapture)
	} else {
		e, err = goaec3.New(p.cfg, p.rate, p.numRender, p.numCapture)
	}
	if err != nil {
		t.Fatal(err)
	}
	cpp := NewEc3Canceller(&p.cfg, mcPtr, p.rate, p.numRender, p.numCapture, ec3Scalar)
	defer cpp.Free()

	fail := func(format string, args ...any) {
		t.Helper()
		def := aec3.DefaultEchoCanceller3Config()
		t.Logf("config:%s", fzConfigDiff(&p.cfg, &def))
		if p.useMC {
			mc := aec3.CreateDefaultMultichannelConfig()
			t.Logf("multichannel config:%s", fzConfigDiff(&p.mc, &mc))
		}
		t.Fatalf(format, args...)
	}
	n := p.rate / 100
	var scene fzAudio = newFzScene(p.seed, p.rate, p.delayMs)
	if p.room {
		scene = newFzRoomAudio(p)
	}
	render := ec3Frame(p.numRender, n)
	capture := ec3Frame(p.numCapture, n)
	linear := ec3Frame(p.numCapture, goaec3.LinearOutputFrameSize)
	cppLin := make([]float32, p.numCapture*goaec3.LinearOutputFrameSize)
	linearEnabled := p.cfg.Filter.ExportLinearAecOutput
	var flat, cppCap []float32
	gainChange := false
	captures := 0

	active := &p.cfg
	if p.useMC && p.numRender > 1 && !p.cfg.MultiChannel.DetectStereoContent {
		active = &p.mc
	}
	var ext *fzExternalDelay
	if p.cfg.Delay.UseExternalDelayEstimator {
		ext = &fzExternalDelay{size: fzRenderDelayBufferSize(active),
			headroomBlocks: active.Delay.DelayHeadroomSamples / aec3.BlockSize}
	}
	stopped := false

	doRender := func(sig int) {
		if stopped {
			return
		}
		if ext != nil {
			ext.render()
		}
		scene.render(render, sig)
		flat = ec3Flat(render, flat)
		cpp.AnalyzeRenderFloat(flat)
		if err := e.AnalyzeRender(render); err != nil {
			t.Fatal(err)
		}
	}
	doCapture := func(sig int, lin bool) {
		if stopped {
			return
		}
		if ext != nil {
			if !ext.captureInRange() {
				stopped = true
				return
			}
			ext.capture()
		}
		scene.capture(capture, sig)
		cppCap = ec3Flat(capture, cppCap)
		withLin := lin && linearEnabled
		if withLin {
			cpp.ProcessCaptureFloat(cppCap, cppLin, gainChange)
			err = e.ProcessCaptureWithLinearOutput(capture, linear)
		} else {
			cpp.ProcessCaptureFloat(cppCap, nil, gainChange)
			err = e.ProcessCapture(capture)
		}
		if err != nil {
			t.Fatal(err)
		}
		gainChange = false
		for ch := range capture {
			for k, v := range capture[ch] {
				if w := cppCap[ch*n+k]; !fzSame32(v, w) {
					fail("%v: capture %d ch %d sample %d: %v want %v", p, captures, ch, k, v, w)
				}
			}
			if withLin {
				for k, v := range linear[ch] {
					if w := cppLin[ch*goaec3.LinearOutputFrameSize+k]; !fzSame32(v, w) {
						fail("%v: capture %d linear ch %d sample %d: %v want %v", p, captures, ch, k, v, w)
					}
				}
			}
		}
		gm, cm := e.Metrics(), cpp.Metrics()
		if !fzSame64(gm.EchoReturnLoss, cm.EchoReturnLoss) ||
			!fzSame64(gm.EchoReturnLossEnhancement, cm.EchoReturnLossEnhancement) || gm.DelayMs != cm.DelayMs {
			fail("%v: capture %d: metrics %+v want %+v", p, captures, gm, cm)
		}
		captures++
	}
	for _, op := range p.ops {
		switch op.kind {
		case fzOpRender:
			doRender(op.sig)
		case fzOpCapture:
			doCapture(op.sig, op.lin)
		case fzOpPairs:
			for range op.n {
				doRender(fzSigEcho)
				doCapture(fzSigEcho, op.lin)
			}
		case fzOpRenderBurst:
			for range op.n {
				doRender(op.sig)
			}
		case fzOpCaptureBurst:
			for range op.n {
				doCapture(op.sig, op.lin)
			}
		case fzOpSetDelay:
			if stopped {
				continue
			}
			if ext != nil {
				ext.setDelay(op.n)
			}
			cpp.SetAudioBufferDelay(op.n)
			e.SetAudioBufferDelay(op.n)
		case fzOpOutputUsage:
			cpp.SetCaptureOutputUsage(op.n != 0)
			e.SetCaptureOutputUsage(op.n != 0)
		case fzOpGainChange:
			e.NotifyEchoPathGainChange()
			gainChange = true
		}
	}
	return captures
}

// fzRandomConfig returns config number i of a series of count: every config
// leaf j with j % count == i takes a random value over its full range (and a
// few other leaves too), then the config is validated and resource-capped.
// Over the series every one of the config leaves is exercised.
func fzRandomConfig(rng *rand.Rand, i, count int) aec3.EchoCanceller3Config {
	for {
		cfg := aec3.DefaultEchoCanceller3Config()
		if rng.Intn(2) == 0 {
			cfg = aec3.CreateDefaultMultichannelConfig()
		}
		base := cfg
		baseLeaves := fzLeaves(&base)
		leaves := fzLeaves(&cfg)
		buf := make([]byte, 7)
		mutate := func(j int) {
			rng.Read(buf)
			buf[0], buf[1] = byte(j), byte(j>>8)
			fzMutate(&fzReader{b: buf}, leaves)
		}
		for range rng.Intn(12) {
			mutate(rng.Intn(len(leaves)))
		}
		for j, l := range leaves {
			if j%count != i {
				continue
			}
			// A designated leaf moves off its default to a value Validate
			// keeps: bools flip, numbers move to a nearby value. The extra
			// mutations above take values over the full ranges.
			switch l.v.Kind() {
			case reflect.Bool:
				l.v.SetBool(!baseLeaves[j].v.Bool())
			case reflect.Float32:
				d := baseLeaves[j].v.Float()
				if d == 0 {
					d = 0.5
				}
				f := d * (0.1 + 0.8*rng.Float64())
				if l.path == "Erle.Min" {
					// Its default is the lower bound.
					f = 1 + 0.4*rng.Float64()
				}
				l.v.SetFloat(float64(float32(f)))
			default:
				if l.path == "Delay.DownSamplingFactor" {
					l.v.SetInt(8)
				} else {
					l.v.SetInt(baseLeaves[j].v.Int() + 1 + int64(rng.Intn(8)))
				}
			}
		}
		cfg.Validate()
		fzCapResources(&cfg)
		if fzDefined(&cfg) {
			return cfg
		}
	}
}

// TestEc3DifferentialRandomConfigs runs long differential sessions of goaec3
// against libwebrtc (60 s of audio each by default, GOAEC3_DIFF_SECONDS
// overrides it; GOAEC3_DIFF_SEED picks another series) with random validated
// configs that together cover every config leaf, random rates, channel
// counts, echo delays and rooms, echo path gain change events, capture output
// usage toggles and short API call bursts.
func TestEc3DifferentialRandomConfigs(t *testing.T) {
	seconds := 60
	if v, err := strconv.Atoi(os.Getenv("GOAEC3_DIFF_SECONDS")); err == nil && v > 0 {
		seconds = v
	}
	count := 8
	if testing.Short() {
		count, seconds = 4, 10
	}
	seed := int64(20261118)
	if v, err := strconv.ParseInt(os.Getenv("GOAEC3_DIFF_SEED"), 10, 64); err == nil {
		seed = v
	}
	rng := rand.New(rand.NewSource(seed))
	def := aec3.DefaultEchoCanceller3Config()
	defLeaves := fzLeaves(&def)
	mutated := make([]bool, len(defLeaves))
	markMutated := func(cfg *aec3.EchoCanceller3Config) {
		for j, l := range fzLeaves(cfg) {
			if fmt.Sprint(l.v.Interface()) != fmt.Sprint(defLeaves[j].v.Interface()) {
				mutated[j] = true
			}
		}
	}
	for i := 0; i < count; i++ {
		p := fzProgram{
			rate:       []int{16000, 32000, 48000}[rng.Intn(3)],
			numRender:  1 + rng.Intn(2),
			numCapture: 1 + rng.Intn(2),
			seed:       rng.Int63n(1 << 30),
			delayMs:    20 + rng.Intn(230),
			room:       true,
		}
		p.cfg = fzRandomConfig(rng, i, count)
		if p.cfg.Delay.UseExternalDelayEstimator && p.numRender > 1 {
			p.cfg.MultiChannel.DetectStereoContent = false
		}
		if rng.Intn(2) == 0 {
			p.useMC = true
			p.mc = fzRandomConfig(rng, (i+1)%count, count)
			fzAgree(&p.mc, &p.cfg)
		}
		if p.cfg.Delay.UseExternalDelayEstimator {
			p.ops = append(p.ops, fzOp{kind: fzOpSetDelay, n: p.delayMs + rng.Intn(40) - 20})
		}
		for frames := 0; frames < seconds*100; {
			switch r := rng.Intn(100); {
			case r < 3:
				p.ops = append(p.ops, fzOp{kind: fzOpGainChange})
			case r < 5:
				p.ops = append(p.ops, fzOp{kind: fzOpOutputUsage, n: rng.Intn(4) & 1})
			case r < 7:
				// A short burst, then the matching frames on the other side.
				k := 1 + rng.Intn(4)
				if rng.Intn(2) == 0 {
					p.ops = append(p.ops, fzOp{kind: fzOpRenderBurst, n: k}, fzOp{kind: fzOpCaptureBurst, n: k, lin: true})
				} else {
					p.ops = append(p.ops, fzOp{kind: fzOpCaptureBurst, n: k, lin: true}, fzOp{kind: fzOpRenderBurst, n: k})
				}
				frames += k
			default:
				k := 1 + rng.Intn(100)
				p.ops = append(p.ops, fzOp{kind: fzOpPairs, n: k, lin: rng.Intn(2) == 0})
				frames += k
			}
		}
		markMutated(&p.cfg)
		if p.useMC {
			markMutated(&p.mc)
		}
		t.Run(fmt.Sprintf("config%d_%dHz_r%d_c%d", i, p.rate, p.numRender, p.numCapture), func(t *testing.T) {
			planned := 0
			for _, op := range p.ops {
				switch op.kind {
				case fzOpPairs, fzOpCaptureBurst:
					planned += op.n
				}
			}
			if done := fzRun(t, &p); done < planned*9/10 {
				t.Errorf("ran %d of %d capture frames", done, planned)
			}
		})
	}
	n := 0
	var untouched []string
	for j, m := range mutated {
		if m {
			n++
		} else {
			untouched = append(untouched, defLeaves[j].path)
		}
	}
	if len(untouched) > 0 && !testing.Short() {
		t.Errorf("config leaves never exercised: %v", untouched)
	}
	t.Logf("%d of %d config leaves differ from the default in at least one validated config; unchanged: %v",
		n, len(mutated), untouched)
}

// TestEc3NaNInputParity feeds one NaN sample to the render or the capture
// signal. FloatToFloatS16 does not clamp NaN; once it reaches the filter
// states both implementations output NaN from the same frame on. The NaN
// bit patterns differ, so NaN outputs compare equal to each other; every
// other sample and the metrics must be bit-exact.
func TestEc3NaNInputParity(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		for _, side := range []string{"render", "capture"} {
			cfg := aec3.DefaultEchoCanceller3Config()
			cfg.Filter.ExportLinearAecOutput = true
			cpp := NewEc3Canceller(&cfg, nil, rate, 1, 2, ec3Scalar)
			e, err := goaec3.New(cfg, rate, 1, 2)
			if err != nil {
				t.Fatal(err)
			}
			n := rate / 100
			scene := newEc3Scene(rate, 1, 2, 60, 4)
			render, capture := ec3Frame(1, n), ec3Frame(2, n)
			lin := ec3Frame(2, goaec3.LinearOutputFrameSize)
			cppLin := make([]float32, 2*160)
			var cppCap []float32
			sawNaN := false
			for frame := 0; frame < 500; frame++ {
				scene.next(render, capture)
				if frame == 200 {
					if side == "render" {
						render[0][17] = float32(math.NaN())
					} else {
						capture[1][17] = float32(math.NaN())
					}
				}
				cpp.AnalyzeRenderFloat(render[0])
				if err := e.AnalyzeRender(render); err != nil {
					t.Fatal(err)
				}
				cppCap = ec3Flat(capture, cppCap)
				cpp.ProcessCaptureFloat(cppCap, cppLin, false)
				if err := e.ProcessCaptureWithLinearOutput(capture, lin); err != nil {
					t.Fatal(err)
				}
				for ch := range capture {
					for k, v := range capture[ch] {
						w := cppCap[ch*n+k]
						sawNaN = sawNaN || v != v
						if !fzSame32(v, w) {
							t.Fatalf("%d Hz %s: frame %d ch %d sample %d: %v want %v", rate, side, frame, ch, k, v, w)
						}
					}
					for k, v := range lin[ch] {
						if w := cppLin[ch*160+k]; !fzSame32(v, w) {
							t.Fatalf("%d Hz %s: frame %d linear ch %d sample %d: %v want %v", rate, side, frame, ch, k, v, w)
						}
					}
				}
				gm, cm := e.Metrics(), cpp.Metrics()
				if !fzSame64(gm.EchoReturnLoss, cm.EchoReturnLoss) ||
					!fzSame64(gm.EchoReturnLossEnhancement, cm.EchoReturnLossEnhancement) || gm.DelayMs != cm.DelayMs {
					t.Fatalf("%d Hz %s: frame %d: metrics %+v want %+v", rate, side, frame, gm, cm)
				}
			}
			cpp.Free()
			// At 32 kHz the two-band split runs on int16 samples, where the
			// NaN is converted to an integer, so only the 16 and 48 kHz
			// capture paths carry it to the output.
			if side == "capture" && rate != 32000 && !sawNaN {
				t.Fatalf("%d Hz %s: the NaN never reached the output", rate, side)
			}
		}
	}
}

// TestEc3UpperBandsGainNaN checks the NaN handling of
// SuppressionGain::UpperBandsGain (suppression_gain.cc), which combines band
// energies and gains with std::max and std::min; those return their first
// argument when the second is NaN. With a zero anti-howling activation
// threshold (the lower Validate bound) the anti-howling gain is sqrtf(0 / 0)
// on silent bands, and with a render power gain of 300 dB the band energies
// overflow to NaN; libwebrtc stays finite at 32 and 48 kHz and the port must
// match it bit for bit.
// TestEc3NaNMinMaxParity covers more rates, configs and inputs.
func TestEc3UpperBandsGainNaN(t *testing.T) {
	cases := map[string]func(*aec3.EchoCanceller3Config){
		"anti-howling threshold 0": func(c *aec3.EchoCanceller3Config) {
			c.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0
		},
		"render power gain 300 dB": func(c *aec3.EchoCanceller3Config) { c.RenderLevels.RenderPowerGainDb = 300 },
	}
	for name, mod := range cases {
		for _, rate := range []int{32000, 48000} {
			t.Run(fmt.Sprintf("%s_%dHz", name, rate), func(t *testing.T) {
				p := fzProgram{rate: rate, numRender: 1, numCapture: 1, cfg: aec3.DefaultEchoCanceller3Config(),
					seed: 1, delayMs: 40, ops: []fzOp{{kind: fzOpPairs, n: 300}}}
				mod(&p.cfg)
				fzRun(t, &p)
			})
		}
	}
}
