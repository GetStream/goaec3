//go:build webrtc_src_oracle && !webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// srcQuietBuffers returns render buffers that the matched filter's
// quiet-render skip accepts for the quiet limit m: every sample at +-m, at
// +m, random magnitudes up to m, and sparse +-m among zeros and subnormals.
func srcQuietBuffers(rng *rand.Rand, size int, m float32) [][]float32 {
	sign := func() float32 {
		if rng.Intn(2) == 0 {
			return -1
		}
		return 1
	}
	out := make([][]float32, 4)
	for i := range out {
		out[i] = make([]float32, size)
	}
	for k := 0; k < size; k++ {
		out[0][k] = sign() * m
		out[1][k] = m
		out[2][k] = sign() * m * float32(rng.Float64())
		switch rng.Intn(3) {
		case 0:
			out[3][k] = sign() * m
		case 1:
			out[3][k] = sign() * 1e-42
		}
	}
	return out
}

// TestSrcMatchedFilterCoreQuietNeverUpdates checks the premise of the
// matched filter's quiet-render skip against the C++ cores of every lane:
// on a render buffer whose samples do not exceed
// aec3.MatchedFilterQuietLimit, no core updates its filter, whatever the
// capture, the start index (including windows that wrap) and the filter;
// and a buffer of constant magnitude 0.1% above the excitation limit does
// update it, so the limit is not vacuous.
func TestSrcMatchedFilterCoreQuietNeverUpdates(t *testing.T) {
	for _, lane := range srcKernLanes(t) {
		t.Run(srcLaneNames[lane], func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(40 + lane)))
			for _, cfg := range []struct {
				hSize int
				limit float32
			}{{512, 150}, {256, 20}, {512, 37.5}, {256, 1024}, {512, 0.37}, {256, 30000}, {512, 1e-6}, {512, 0}} {
				m, ok := aec3.MatchedFilterQuietLimit(cfg.hSize, cfg.limit)
				if !ok {
					t.Fatalf("no quiet limit for %+v", cfg)
				}
				thr := float32(float32(cfg.hSize)*cfg.limit) * cfg.limit
				xSize := cfg.hSize/32*153 + rng.Intn(3)
				for bi, x := range srcQuietBuffers(rng, xSize, m) {
					for _, accum := range []bool{false, true} {
						for call := 0; call < 150; call++ {
							y := srcSpeechLike(rng, cfg.hSize/32, 5000)
							xs := rng.Intn(xSize)
							if call%2 == 0 {
								xs = xSize - 1 - rng.Intn(cfg.hSize+20)
							}
							h0 := srcSpeechLike(rng, cfg.hSize, 0.05)
							h := append([]float32(nil), h0...)
							acc := make([]float32, cfg.hSize/4)
							scr := make([]float32, cfg.hSize)
							var upd bool
							var es float32
							SrcMatchedFilterCore(lane, xs, thr, 0.7, x, y, h, &upd, &es, accum, acc, scr)
							what := fmt.Sprintf("%+v buffer %d accum=%v xs=%d", cfg, bi, accum, xs)
							if upd {
								t.Fatalf("%s: the C++ core updated the filter on quiet render", what)
							}
							srcEqualBits(t, what+" h", h, h0)
						}
					}
				}
				if cfg.limit == 0 {
					continue
				}
				// Just above the excitation limit the cores do update.
				x := make([]float32, xSize)
				for k := range x {
					x[k] = cfg.limit * 1.001
				}
				for _, accum := range []bool{false, true} {
					y := srcSpeechLike(rng, cfg.hSize/32, 5000)
					h := make([]float32, cfg.hSize)
					acc := make([]float32, cfg.hSize/4)
					scr := make([]float32, cfg.hSize)
					var upd bool
					var es float32
					SrcMatchedFilterCore(lane, 7, thr, 0.7, x, y, h, &upd, &es, accum, acc, scr)
					if !upd {
						t.Fatalf("%+v accum=%v: no update just above the excitation limit", cfg, accum)
					}
				}
			}
		})
	}
}
