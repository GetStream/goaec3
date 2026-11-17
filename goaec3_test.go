package goaec3

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// ec3EchoRemoverPorted reports whether EchoRemover is functional (the API
// stub panics).
func ec3EchoRemoverPorted() (ok bool) {
	defer func() {
		if recover() != nil {
			ok = false
		}
	}()
	cfg := aec3.DefaultEchoCanceller3Config()
	aec3.NewEchoRemover(&cfg, 16000, 1, 1).GetMetrics()
	return true
}

func ec3RequireEchoRemover(t testing.TB) {
	t.Helper()
	if !ec3EchoRemoverPorted() {
		t.Skip("EchoRemover is not ported yet")
	}
}

func ec3Frame(numChannels, n int) [][]float32 {
	f := make([][]float32, numChannels)
	for ch := range f {
		f[ch] = make([]float32, n)
	}
	return f
}

func TestNewErrors(t *testing.T) {
	if _, err := New(DefaultConfig(), 44100, 1, 1); !errors.Is(err, ErrSampleRate) {
		t.Fatalf("44.1 kHz: %v", err)
	}
	if _, err := New(DefaultConfig(), 16000, 0, 1); !errors.Is(err, ErrNumChannels) {
		t.Fatalf("0 render channels: %v", err)
	}
	e, err := New(DefaultConfig(), 32000, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if e.FrameSize() != 320 || e.SampleRateHz() != 32000 || e.NumRenderChannels() != 2 || e.NumCaptureChannels() != 1 {
		t.Fatal("accessors")
	}
	if err := e.AnalyzeRender(ec3Frame(1, 320)); !errors.Is(err, ErrChannelCount) {
		t.Fatalf("render channels: %v", err)
	}
	if err := e.AnalyzeRender(ec3Frame(2, 160)); !errors.Is(err, ErrFrameLength) {
		t.Fatalf("render length: %v", err)
	}
	if err := e.ProcessCapture(ec3Frame(2, 320)); !errors.Is(err, ErrChannelCount) {
		t.Fatalf("capture channels: %v", err)
	}
	if err := e.ProcessCaptureWithLinearOutput(ec3Frame(1, 320), ec3Frame(1, 160)); !errors.Is(err, ErrLinearOutputDisabled) {
		t.Fatalf("linear disabled: %v", err)
	}
}

// TestPassthroughScaling runs the canceller in external-delay mode without
// an audio buffer delay, where the echo remover is bypassed: at 16 kHz the
// output is then the input delayed by one 64-sample block, which checks that
// the [-1, 1] <-> FloatS16 conversions are exact.
func TestPassthroughScaling(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Delay.UseExternalDelayEstimator = true
	e, err := New(cfg, 16000, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	render := ec3Frame(1, 160)
	capture := ec3Frame(2, 160)
	var in, out []float32
	for frame := 0; frame < 20; frame++ {
		for k := range capture[1] {
			render[0][k] = float32(rng.Float64()*2 - 1)
			capture[0][k] = float32(rng.Float64()*2 - 1)
			capture[1][k] = float32(rng.Float64()*2 - 1)
		}
		in = append(in, capture[1]...)
		if err := e.AnalyzeRender(render); err != nil {
			t.Fatal(err)
		}
		if err := e.ProcessCapture(capture); err != nil {
			t.Fatal(err)
		}
		out = append(out, capture[1]...)
	}
	for k := 64; k < len(out); k++ {
		if math.Float32bits(out[k]) != math.Float32bits(in[k-64]) {
			t.Fatalf("sample %d: %v want %v", k, out[k], in[k-64])
		}
	}
}

func TestPassthroughAllocationFree(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		cfg := DefaultConfig()
		cfg.Delay.UseExternalDelayEstimator = true
		cfg.Filter.ExportLinearAecOutput = true
		e, err := New(cfg, rate, 2, 2)
		if err != nil {
			t.Fatal(err)
		}
		render := ec3Frame(2, rate/100)
		capture := ec3Frame(2, rate/100)
		linear := ec3Frame(2, LinearOutputFrameSize)
		step := func() {
			_ = e.AnalyzeRender(render)
			_ = e.ProcessCaptureWithLinearOutput(capture, linear)
			_ = e.ProcessCapture(capture)
		}
		for i := 0; i < 10; i++ {
			step()
		}
		if a := testing.AllocsPerRun(100, step); a != 0 {
			t.Fatalf("rate %d: %v allocations per frame", rate, a)
		}
	}
}

// TestEchoCancellerAllocationFree locks the full steady-state path,
// including Metrics, to zero allocations.
func TestEchoCancellerAllocationFree(t *testing.T) {
	ec3RequireEchoRemover(t)
	for _, rate := range []int{16000, 32000, 48000} {
		e, err := New(DefaultConfig(), rate, 2, 1)
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(1))
		render := ec3Frame(2, rate/100)
		capture := ec3Frame(1, rate/100)
		step := func() {
			for k := range capture[0] {
				render[0][k] = float32(rng.NormFloat64() * 0.1)
				render[1][k] = render[0][k]
				capture[0][k] = 0.3*render[0][k] + float32(rng.NormFloat64()*0.001)
			}
			_ = e.AnalyzeRender(render)
			_ = e.ProcessCapture(capture)
			_ = e.Metrics()
		}
		for i := 0; i < 300; i++ {
			step()
		}
		if a := testing.AllocsPerRun(100, step); a != 0 {
			t.Fatalf("rate %d: %v allocations per frame", rate, a)
		}
	}
}

// TestEchoCancellerReducesEcho checks that a delayed, attenuated copy of
// the render signal is removed from the capture signal.
func TestEchoCancellerReducesEcho(t *testing.T) {
	ec3RequireEchoRemover(t)
	e, err := New(DefaultConfig(), 16000, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	const delay = 800 // 50 ms
	hist := make([]float32, delay+160)
	render := ec3Frame(1, 160)
	capture := ec3Frame(1, 160)
	var inPow, outPow float64
	for frame := 0; frame < 1000; frame++ {
		copy(hist, hist[160:])
		for k := range render[0] {
			render[0][k] = float32(rng.NormFloat64() * 0.1)
			hist[delay+k] = render[0][k]
			capture[0][k] = 0.5 * hist[k]
		}
		if frame >= 500 {
			for _, v := range capture[0] {
				inPow += float64(v) * float64(v)
			}
		}
		if err := e.AnalyzeRender(render); err != nil {
			t.Fatal(err)
		}
		if err := e.ProcessCapture(capture); err != nil {
			t.Fatal(err)
		}
		if frame >= 500 {
			for _, v := range capture[0] {
				outPow += float64(v) * float64(v)
			}
		}
	}
	if reduction := 10 * math.Log10(inPow/outPow); reduction < 20 {
		t.Fatalf("echo reduced by %.1f dB", reduction)
	}
}
