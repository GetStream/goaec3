//go:build webrtc_oracle || webrtc_src_oracle

package webrtcoracle

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/GetStream/goaec3"
	"github.com/GetStream/goaec3/internal/aec3"
)

// TestEc3CoarseFilterFillsRenderBuffer runs a coarse filter of as many
// blocks as the render buffer has, the longest New accepts next to a given
// refined filter: every partition loop wraps around the buffer at each
// position of it. One block more is rejected: the C++ SIMD lanes then read
// past the buffer.
func TestEc3CoarseFilterFillsRenderBuffer(t *testing.T) {
	for _, c := range []struct {
		numFilters, refined int
		external            bool
	}{{5, 13, false}, {5, 13, true}, {1, 9, false}, {2, 20, true}} {
		t.Run(fmt.Sprintf("filters%d_refined%d_external%v", c.numFilters, c.refined, c.external), func(t *testing.T) {
			cfg := goaec3.DefaultConfig()
			cfg.Delay.NumFilters = c.numFilters
			cfg.Delay.UseExternalDelayEstimator = c.external
			cfg.Filter.Refined.LengthBlocks = c.refined
			cfg.Filter.RefinedInitial.LengthBlocks = min(cfg.Filter.RefinedInitial.LengthBlocks, c.refined)
			cfg.Filter.ExportLinearAecOutput = true
			size := aec3.GetRenderDelayBufferSize(cfg.Delay.DownSamplingFactor, c.numFilters, c.refined)
			cfg.Filter.Coarse.LengthBlocks = size + 1
			if _, err := goaec3.New(cfg, 16000, 2, 1); !errors.Is(err, goaec3.ErrConfig) {
				t.Fatalf("coarse filter of %d blocks: New error = %v, want ErrConfig", size+1, err)
			}
			cfg.Filter.Coarse.LengthBlocks = size
			cpp := NewEc3Canceller(&cfg, nil, 16000, 2, 1, ec3Scalar)
			defer cpp.Free()
			e, err := goaec3.New(cfg, 16000, 2, 1)
			if err != nil {
				t.Fatal(err)
			}
			if c.external {
				cpp.SetAudioBufferDelay(40)
				e.SetAudioBufferDelay(40)
			}
			scene := newEc3Scene(16000, 2, 1, 100, 3)
			render, capture := ec3Frame(2, 160), ec3Frame(1, 160)
			lin, cppLin := ec3Frame(1, 160), make([]float32, 160)
			var flat, cppCap []float32
			frames := 3 * size
			if testing.Short() {
				frames = size
			}
			for frame := range frames {
				scene.next(render, capture)
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
				for k := range capture[0] {
					if math.Float32bits(capture[0][k]) != math.Float32bits(cppCap[k]) {
						t.Fatalf("frame %d sample %d: %v want %v", frame, k, capture[0][k], cppCap[k])
					}
					if math.Float32bits(lin[0][k]) != math.Float32bits(cppLin[k]) {
						t.Fatalf("frame %d linear sample %d: %v want %v", frame, k, lin[0][k], cppLin[k])
					}
				}
			}
		})
	}
}
