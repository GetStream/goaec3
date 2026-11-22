//go:build webrtc_oracle || webrtc_src_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3"
	"github.com/GetStream/goaec3/internal/aec3"
)

// ec3BenchSignal returns numFrames 10 ms frames of interleaved-by-channel
// ([frame][ch*n + i]) render and capture audio in [-1, 1]: speech-like
// modulated noise on the render side, and capture = render through a ~100 ms
// decaying room response plus near-end noise.
func ec3BenchSignal(rate, numCh, numFrames int) (render, capture [][]float32) {
	n := rate / 100
	rng := rand.New(rand.NewSource(7))
	total := numFrames * n
	x := make([][]float32, numCh)
	for ch := range x {
		x[ch] = make([]float32, total)
		env := 0.0
		for i := range x[ch] {
			if i%(rate/8) == 0 {
				env = 0.05 + 0.25*rng.Float64()
				if rng.Intn(4) == 0 {
					env = 0.002
				}
			}
			x[ch][i] = float32(env * rng.NormFloat64() * (0.6 + 0.4*math.Sin(float64(i)*2*math.Pi*3/float64(rate))))
		}
	}
	delay := rate / 10
	taps := rate / 50
	h := make([]float32, taps)
	for k := range h {
		h[k] = float32(0.4 * math.Exp(-float64(k)/float64(taps/6)) * rng.NormFloat64())
	}
	render = make([][]float32, numFrames)
	capture = make([][]float32, numFrames)
	for f := range numFrames {
		render[f] = make([]float32, numCh*n)
		capture[f] = make([]float32, numCh*n)
		for ch := range numCh {
			for i := range n {
				t := f*n + i
				render[f][ch*n+i] = x[ch][t]
				var y float32
				for k := range taps {
					if s := t - delay - k; s >= 0 {
						y += h[k] * x[0][s]
					}
				}
				capture[f][ch*n+i] = y + float32(0.002*rng.NormFloat64())
			}
		}
	}
	return render, capture
}

func ec3SplitFrame(frame []float32, numCh int, dst [][]float32) {
	n := len(frame) / numCh
	for ch := range numCh {
		copy(dst[ch], frame[ch*n:(ch+1)*n])
	}
}

// ec3BenchScenario returns the bench signal with one side replaced, so the
// states a real call spends most of its time in are timed as well:
//
//   - "renderZero": digital silence on the render side, near-end noise on
//     the capture side;
//   - "renderQuiet": comfort noise at about -70 dBFS on the render side,
//     near-end noise on the capture side;
//   - "captureZero": the bench render signal, digital silence on the capture
//     side.
//
// The empty scenario is the bench signal itself.
func ec3BenchScenario(scenario string, render0, capture0 [][]float32) (render, capture [][]float32) {
	if scenario == "" {
		return render0, capture0
	}
	rng := rand.New(rand.NewSource(11))
	render = make([][]float32, len(render0))
	capture = make([][]float32, len(capture0))
	for f := range render0 {
		render[f] = append([]float32(nil), render0[f]...)
		capture[f] = append([]float32(nil), capture0[f]...)
		switch scenario {
		case "renderZero":
			clear(render[f])
			for i := range capture[f] {
				capture[f][i] = float32(0.05 * rng.NormFloat64())
			}
		case "renderQuiet":
			for i := range render[f] {
				render[f][i] = float32(0.0003 * rng.NormFloat64())
			}
			for i := range capture[f] {
				capture[f][i] = float32(0.05 * rng.NormFloat64())
			}
		case "captureZero":
			clear(capture[f])
		default:
			panic("unknown scenario " + scenario)
		}
	}
	return render, capture
}

// BenchmarkEc3VsLibwebrtc reports ns per 10 ms frame (AnalyzeRender +
// ProcessCapture) for the Go public API and for libwebrtc's EchoCanceller3
// through AudioBuffer, on identical input: the converged bench signal
// (cases named <rate>_<channels>) and the scenarios of ec3BenchScenario
// (cases named <rate>_<channels>_<scenario>).
func BenchmarkEc3VsLibwebrtc(b *testing.B) {
	const warmup = 600
	const frames = 1600
	type benchCase struct {
		rate, ch int
		scenario string
	}
	cases := []benchCase{{16000, 1, ""}, {48000, 1, ""}, {48000, 2, ""}}
	for _, tc := range []benchCase{{16000, 1, ""}, {48000, 2, ""}} {
		for _, sc := range []string{"renderZero", "renderQuiet", "captureZero"} {
			cases = append(cases, benchCase{tc.rate, tc.ch, sc})
		}
	}
	type sigKey struct{ rate, ch int }
	type sig struct{ render, capture [][]float32 }
	signals := map[sigKey]sig{}
	for _, tc := range cases {
		key := sigKey{tc.rate, tc.ch}
		base, ok := signals[key]
		if !ok {
			base.render, base.capture = ec3BenchSignal(tc.rate, tc.ch, frames)
			signals[key] = base
		}
		render, capture := ec3BenchScenario(tc.scenario, base.render, base.capture)
		name := fmt.Sprintf("%dkHz_%dch", tc.rate/1000, tc.ch)
		if tc.scenario != "" {
			name += "_" + tc.scenario
		}
		b.Run(name+"/go", func(b *testing.B) {
			ec, err := goaec3.New(goaec3.DefaultConfig(), tc.rate, tc.ch, tc.ch)
			if err != nil {
				b.Fatal(err)
			}
			n := tc.rate / 100
			r := make([][]float32, tc.ch)
			c := make([][]float32, tc.ch)
			for ch := range tc.ch {
				r[ch] = make([]float32, n)
				c[ch] = make([]float32, n)
			}
			step := func(f int) {
				ec3SplitFrame(render[f], tc.ch, r)
				ec3SplitFrame(capture[f], tc.ch, c)
				_ = ec.AnalyzeRender(r)
				_ = ec.ProcessCapture(c)
			}
			for f := range warmup {
				step(f)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				step(warmup + i%(frames-warmup))
			}
		})
		b.Run(name+"/libwebrtc", func(b *testing.B) {
			cfg := aec3.DefaultEchoCanceller3Config()
			ec := NewEc3Canceller(&cfg, nil, tc.rate, tc.ch, tc.ch, false)
			defer ec.Free()
			buf := make([]float32, len(capture[0]))
			step := func(f int) {
				ec.AnalyzeRenderFloat(render[f])
				copy(buf, capture[f])
				ec.ProcessCaptureFloat(buf, nil, false)
			}
			for f := range warmup {
				step(f)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				step(warmup + i%(frames-warmup))
			}
		})
	}
}
