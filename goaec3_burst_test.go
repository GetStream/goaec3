package goaec3

import (
	"math/rand"
	"testing"
)

// TestExternalDelayRenderBurstsDoNotPanic drives the external-delay mode with
// render bursts long enough that the C++ computes a block read index outside
// the render ring (undefined behavior there). The port must keep running.
func TestExternalDelayRenderBurstsDoNotPanic(t *testing.T) {
	for _, rate := range []int{16000, 48000} {
		cfg := DefaultConfig()
		cfg.Delay.UseExternalDelayEstimator = true
		ec, err := New(cfg, rate, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		ec.SetAudioBufferDelay(120)
		n := ec.FrameSize()
		render := [][]float32{make([]float32, n)}
		capture := [][]float32{make([]float32, n)}
		rng := rand.New(rand.NewSource(1))
		for round := 0; round < 40; round++ {
			for range 50 + rng.Intn(60) {
				for i := range render[0] {
					render[0][i] = float32(rng.NormFloat64() * 0.1)
				}
				if err := ec.AnalyzeRender(render); err != nil {
					t.Fatal(err)
				}
			}
			for range 1 + rng.Intn(3) {
				for i := range capture[0] {
					capture[0][i] = float32(rng.NormFloat64() * 0.05)
				}
				if err := ec.ProcessCapture(capture); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
