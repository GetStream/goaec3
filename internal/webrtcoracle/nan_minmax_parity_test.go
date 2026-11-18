//go:build webrtc_oracle || webrtc_src_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/GetStream/goaec3"
	"github.com/GetStream/goaec3/internal/aec3"
)

// TestEc3NaNMinMaxParity runs EchoCanceller3 end to end against libwebrtc
// with configs and inputs that put NaN, Inf and signed zeros into the float
// min/max/clamp sites the port reproduces with the C++ semantics (std::min,
// std::max, std::minmax_element, SafeClamp, fabsf and the per-lane SIMD
// max):
//
//   - Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0:
//     SuppressionGain::UpperBandsGain computes the anti-howling gain as
//     sqrtf(0 / 0) = NaN on silent bands, which std::min drops;
//   - RenderLevels.RenderPowerGainDb of 150, 250 and 300 dB: the render
//     signal reaches 1e12 to 1e20, its power overflows to Inf and the band
//     energies, spectra and filter states turn to Inf and NaN;
//   - render and capture that alternate between echo and digital silence,
//     full-scale and +-Inf samples (clamped by FloatToFloatS16), -0 samples,
//     and a NaN sample in the render or the capture signal.
//
// Every capture output sample, linear output sample and metric must be
// bit-exact, except that where libwebrtc itself outputs a NaN the port must
// output a NaN with any payload (NaN payloads are not part of IEEE 754
// arithmetic results). The run logs where libwebrtc produces NaN.
func TestEc3NaNMinMaxParity(t *testing.T) {
	type mod struct {
		name string
		f    func(*aec3.EchoCanceller3Config)
	}
	mods := []mod{
		{"antihowling0", func(c *aec3.EchoCanceller3Config) {
			c.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0
		}},
		{"gain150", func(c *aec3.EchoCanceller3Config) { c.RenderLevels.RenderPowerGainDb = 150 }},
		{"gain250", func(c *aec3.EchoCanceller3Config) { c.RenderLevels.RenderPowerGainDb = 250 }},
		{"gain300", func(c *aec3.EchoCanceller3Config) { c.RenderLevels.RenderPowerGainDb = 300 }},
		{"antihowling0_gain300", func(c *aec3.EchoCanceller3Config) {
			c.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0
			c.RenderLevels.RenderPowerGainDb = 300
		}},
	}
	scenarios := []string{"silence", "edges", "nanrender", "nancapture"}
	frames := 600
	if testing.Short() {
		frames = 300
	}
	var cppNaN []string
	for _, m := range mods {
		for _, rate := range []int{16000, 32000, 48000} {
			for _, chs := range [][2]int{{1, 1}, {2, 2}} {
				for _, sc := range scenarios {
					name := fmt.Sprintf("%s/%dHz/%dx%d/%s", m.name, rate, chs[0], chs[1], sc)
					t.Run(name, func(t *testing.T) {
						cfg := aec3.DefaultEchoCanceller3Config()
						cfg.Filter.ExportLinearAecOutput = true
						m.f(&cfg)
						if !cfg.Validate() {
							t.Fatal("config does not validate")
						}
						if first := nanMinMaxRun(t, &cfg, rate, chs[0], chs[1], sc, frames); first >= 0 {
							cppNaN = append(cppNaN, fmt.Sprintf("%s (from capture frame %d)", name, first))
						}
					})
				}
			}
		}
	}
	sort.Strings(cppNaN)
	t.Logf("libwebrtc outputs NaN in %d of %d runs:\n%s", len(cppNaN),
		len(mods)*3*2*len(scenarios), strings.Join(cppNaN, "\n"))
}

// nanMinMaxSame reports whether the port's value got matches libwebrtc's
// want: bit-exact, or both NaN where libwebrtc outputs NaN.
func nanMinMaxSame(got, want float32) bool {
	return math.Float32bits(got) == math.Float32bits(want) || (want != want && got != got)
}

// nanMinMaxRun drives one scenario and returns the first capture frame in
// which libwebrtc outputs a NaN, or -1.
func nanMinMaxRun(t *testing.T, cfg *aec3.EchoCanceller3Config, rate, numRender, numCapture int,
	scenario string, frames int) int {
	t.Helper()
	e, err := goaec3.New(*cfg, rate, numRender, numCapture)
	if err != nil {
		t.Fatal(err)
	}
	cpp := NewEc3Canceller(cfg, nil, rate, numRender, numCapture, ec3Scalar)
	defer cpp.Free()

	n := rate / 100
	scene := newEc3Scene(rate, numRender, numCapture, 40, int64(rate+numRender))
	render, capture := ec3Frame(numRender, n), ec3Frame(numCapture, n)
	lin := ec3Frame(numCapture, goaec3.LinearOutputFrameSize)
	cppLin := make([]float32, numCapture*goaec3.LinearOutputFrameSize)
	var flat, cppCap []float32
	negZero := float32(math.Copysign(0, -1))
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	firstNaN := -1
	set := func(f [][]float32, v float32) {
		for ch := range f {
			for k := range f[ch] {
				f[ch][k] = v
			}
		}
	}
	for frame := 0; frame < frames; frame++ {
		scene.next(render, capture)
		// Silent stretches with echo in between: the band energies are
		// exactly zero, then large again.
		silent := frame < 60 || (frame/40)%3 == 1
		switch scenario {
		case "silence":
			if silent {
				set(render, 0)
				set(capture, 0)
			}
		case "edges":
			switch (frame / 25) % 6 {
			case 1:
				set(render, 0)
				set(capture, negZero)
			case 2:
				for ch := range render {
					for k := range render[ch] {
						render[ch][k] = float32(1 - 2*(k&1))
					}
				}
			case 3:
				render[0][k3(frame, n)] = inf
				render[numRender-1][k3(frame+1, n)] = -inf
				capture[0][k3(frame+2, n)] = inf
			case 4:
				set(render, negZero)
			}
		case "nanrender":
			if silent {
				set(render, 0)
				set(capture, 0)
			}
			if frame == frames/2 {
				render[numRender-1][k3(frame, n)] = nan
			}
		case "nancapture":
			if silent {
				set(render, 0)
				set(capture, 0)
			}
			if frame == frames/2 {
				capture[numCapture-1][k3(frame, n)] = nan
			}
		}

		flat = ec3Flat(render, flat)
		cpp.AnalyzeRenderFloat(flat)
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
				if w != w && firstNaN < 0 {
					firstNaN = frame
				}
				if !nanMinMaxSame(v, w) {
					t.Fatalf("frame %d ch %d sample %d: %v (%#x) want %v (%#x)", frame, ch, k,
						v, math.Float32bits(v), w, math.Float32bits(w))
				}
			}
			for k, v := range lin[ch] {
				w := cppLin[ch*goaec3.LinearOutputFrameSize+k]
				if !nanMinMaxSame(v, w) {
					t.Fatalf("frame %d linear ch %d sample %d: %v (%#x) want %v (%#x)", frame, ch, k,
						v, math.Float32bits(v), w, math.Float32bits(w))
				}
			}
		}
		gm, cm := e.Metrics(), cpp.Metrics()
		same64 := func(a, b float64) bool {
			return math.Float64bits(a) == math.Float64bits(b) || (b != b && a != a)
		}
		if !same64(gm.EchoReturnLoss, cm.EchoReturnLoss) ||
			!same64(gm.EchoReturnLossEnhancement, cm.EchoReturnLossEnhancement) || gm.DelayMs != cm.DelayMs {
			t.Fatalf("frame %d: metrics %+v want %+v", frame, gm, cm)
		}
	}
	return firstNaN
}

// k3 picks a sample index from the frame number.
func k3(frame, n int) int { return (frame*37 + 11) % n }
