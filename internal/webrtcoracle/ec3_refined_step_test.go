//go:build webrtc_oracle || webrtc_src_oracle

package webrtcoracle

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3"
	"github.com/GetStream/goaec3/internal/aec3"
)

// TestEc3RefinedStepNaN: with a refined filter ErrorFloor, ErrorCeil or
// NoiseGate of 0, or a denormal one, all of which Validate accepts,
// libwebrtc turns finite input into NaN. The step size of the filter is
// h / (0.5*h*X2 + n*E2) for X2 >= NoiseGate and h clamped to ErrorFloor from
// below, then to ErrorCeil from above (refined_filter_update_gain.cc); under
// digital silence in the capture signal E2 is 0, so the step size is 0/0
// for h = 0 or X2 = 0 and infinite once 0.5*h underflows. The filter is NaN
// from then on. New rejects these configs.
func TestEc3RefinedStepNaN(t *testing.T) {
	for _, c := range []struct {
		name   string
		render bool // noise in the render signal; digital silence otherwise
		set    func(*aec3.EchoCanceller3Config)
	}{
		{"RefinedInitial.ErrorFloor=0", true, func(c *aec3.EchoCanceller3Config) { c.Filter.RefinedInitial.ErrorFloor = 0 }},
		{"RefinedInitial.ErrorFloor=1e-45", true, func(c *aec3.EchoCanceller3Config) {
			c.Filter.RefinedInitial.ErrorFloor = 1e-45
		}},
		{"Refined.ErrorFloor=0", true, func(c *aec3.EchoCanceller3Config) {
			c.Filter.Refined.ErrorFloor = 0
			c.Filter.InitialStateSeconds = 0
		}},
		{"RefinedInitial.ErrorCeil=0", true, func(c *aec3.EchoCanceller3Config) { c.Filter.RefinedInitial.ErrorCeil = 0 }},
		{"RefinedInitial.NoiseGate=0", false, func(c *aec3.EchoCanceller3Config) { c.Filter.RefinedInitial.NoiseGate = 0 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := aec3.DefaultEchoCanceller3Config()
			c.set(&cfg)
			cfg.Filter.ExportLinearAecOutput = true
			if v := cfg; !v.Validate() {
				t.Fatal("Validate changes the config")
			}
			if _, err := goaec3.New(cfg, 16000, 1, 1); !errors.Is(err, goaec3.ErrConfig) {
				t.Fatalf("New error = %v, want ErrConfig", err)
			}
			cpp := NewEc3Canceller(&cfg, nil, 16000, 1, 1, ec3Scalar)
			defer cpp.Free()
			rng := rand.New(rand.NewSource(1))
			render, capture, lin := make([]float32, 160), make([]float32, 160), make([]float32, 160)
			for frame := range 1500 {
				if c.render {
					for k := range render {
						render[k] = float32(0.2 * rng.NormFloat64())
					}
				}
				cpp.AnalyzeRenderFloat(render)
				clear(capture)
				cpp.ProcessCaptureFloat(capture, lin, false)
				for _, v := range lin {
					if v != v {
						t.Logf("libwebrtc: NaN linear output at capture frame %d", frame)
						return
					}
				}
			}
			t.Fatal("libwebrtc kept its output finite")
		})
	}
}

// TestEc3CoarseStepNaN: the step size of the coarse filter is Rate / X2 for
// a render power X2 above NoiseGate (coarse_filter_update_gain.cc). With a
// gate of 0, which Validate accepts, a render signal in the subnormal range
// behind the high-pass filter has a subnormal X2: the step size is infinite
// and the output is NaN from then on. New rejects the config.
func TestEc3CoarseStepNaN(t *testing.T) {
	cfg := aec3.DefaultEchoCanceller3Config()
	cfg.Filter.CoarseInitial.NoiseGate = 0
	cfg.Filter.HighPassFilterEchoReference = true
	if v := cfg; !v.Validate() {
		t.Fatal("Validate changes the config")
	}
	if _, err := goaec3.New(cfg, 16000, 1, 1); !errors.Is(err, goaec3.ErrConfig) {
		t.Fatalf("New error = %v, want ErrConfig", err)
	}
	cpp := NewEc3Canceller(&cfg, nil, 16000, 1, 1, ec3Scalar)
	defer cpp.Free()
	rng := rand.New(rand.NewSource(1))
	render, capture := make([]float32, 160), make([]float32, 160)
	subnormal := func() float32 {
		return math.Float32frombits(uint32(rng.Intn(1<<23)) | uint32(rng.Intn(2))<<31)
	}
	var delayed [4][160]float32 // the last render frames, for a 40 ms echo
	for frame := range 450 {
		// Silence, noise, noise with its echo, faint noise, then subnormal
		// samples on both sides.
		copy(delayed[:], delayed[1:])
		for k := range render {
			render[k], capture[k] = 0, 0
			switch {
			case frame < 30:
			case frame < 120:
				render[k] = float32(0.2 * rng.NormFloat64())
			case frame < 300:
				render[k] = float32(0.2 * rng.NormFloat64())
				capture[k] = 0.5*delayed[0][k] + float32(0.001*rng.NormFloat64())
			case frame < 330:
				render[k] = float32(1e-6 * rng.NormFloat64())
			default:
				render[k], capture[k] = subnormal(), subnormal()
			}
		}
		delayed[3] = [160]float32(render)
		cpp.AnalyzeRenderFloat(render)
		cpp.ProcessCaptureFloat(capture, nil, false)
		for _, v := range capture {
			if v != v {
				t.Logf("libwebrtc: NaN output at capture frame %d", frame)
				return
			}
		}
	}
	t.Fatal("libwebrtc kept its output finite")
}
