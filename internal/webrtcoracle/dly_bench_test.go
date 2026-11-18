//go:build webrtc_oracle

package webrtcoracle

import (
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// BenchmarkDlyMatchedFilterCoreVsLibwebrtc times one MatchedFilterCore call
// (16 capture samples against a 512-tap filter, the default configuration)
// in Go and in libwebrtc. The filter is restored before every call so both
// sides do the same work; updating cases update on every sample.
func BenchmarkDlyMatchedFilterCoreVsLibwebrtc(b *testing.B) {
	rng := rand.New(rand.NewSource(11))
	const hSize = 512
	xSize := aec3.GetDownSampledBufferSize(4, 5)
	x := dlySpeechLike(rng, xSize, 5000)
	y := dlySpeechLike(rng, 16, 5000)
	h0 := make([]float32, hSize)
	for i := range h0 {
		h0[i] = float32(rng.NormFloat64() * 0.01)
	}
	for _, tc := range []struct {
		name  string
		lane  DlyMatchedFilterCoreLane
		accum bool
		thr   float32
		xs    int
	}{
		{"neon", DlyCoreNeon, false, 1, 100},
		{"neon_wrap", DlyCoreNeon, false, 1, xSize - 300},
		{"neon_noupdate", DlyCoreNeon, false, 1e30, 100},
		{"neon_accerr", DlyCoreNeonAccumulatedError, true, 1, 100},
		{"scalar", DlyCoreScalar, false, 1, 100},
	} {
		h := make([]float32, hSize)
		acc := make([]float32, hSize/4)
		scr := make([]float32, hSize)
		b.Run(tc.name+"/go", func(b *testing.B) {
			var upd bool
			var es float32
			for i := 0; i < b.N; i++ {
				copy(h, h0)
				switch tc.lane {
				case DlyCoreScalar:
					aec3.MatchedFilterCore(tc.xs, tc.thr, 0.7, x, y, h, &upd, &es, tc.accum, acc)
				case DlyCoreNeonAccumulatedError:
					aec3.MatchedFilterCoreWithAccumulatedErrorNeon(tc.xs, tc.thr, 0.7, x, y, h, &upd, &es, acc, scr)
				default:
					aec3.MatchedFilterCoreNeon(tc.xs, tc.thr, 0.7, x, y, h, &upd, &es, tc.accum, acc, scr)
				}
			}
		})
		b.Run(tc.name+"/libwebrtc", func(b *testing.B) {
			var upd bool
			var es float32
			for i := 0; i < b.N; i++ {
				copy(h, h0)
				DlyMatchedFilterCore(tc.lane, tc.xs, tc.thr, 0.7, x, y, h, &upd, &es, tc.accum, acc, scr)
			}
		})
	}
}
