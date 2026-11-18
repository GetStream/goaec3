//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// dlyQuietBuffers returns render buffers that the matched filter's
// quiet-render skip accepts for the quiet limit m: every sample at +-m, at
// +m, random magnitudes up to m, and sparse +-m among zeros and subnormals.
func dlyQuietBuffers(rng *rand.Rand, size int, m float32) [][]float32 {
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

// TestDlyMatchedFilterCoreQuietNeverUpdates checks the premise of the
// matched filter's quiet-render skip against the C++ cores of libwebrtc
// (scalar, NEON, NEON with accumulated error): on a render buffer whose
// samples do not exceed aec3.MatchedFilterQuietLimit, no core updates its
// filter, whatever the capture, the start index (including windows that
// wrap) and the filter; and a buffer of constant magnitude 0.1% above the
// excitation limit does update it, so the limit is not vacuous.
func TestDlyMatchedFilterCoreQuietNeverUpdates(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	type lane struct {
		lane  DlyMatchedFilterCoreLane
		accum bool
	}
	lanes := []lane{{DlyCoreScalar, false}, {DlyCoreScalar, true}, {DlyCoreNeon, false}, {DlyCoreNeon, true},
		{DlyCoreNeonAccumulatedError, true}}
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
		for bi, x := range dlyQuietBuffers(rng, xSize, m) {
			for _, l := range lanes {
				for call := 0; call < 150; call++ {
					y := dlySpeechLike(rng, cfg.hSize/32, 5000)
					xs := rng.Intn(xSize)
					if call%2 == 0 {
						xs = xSize - 1 - rng.Intn(cfg.hSize+20)
					}
					h0 := dlySpeechLike(rng, cfg.hSize, 0.05)
					h := append([]float32(nil), h0...)
					acc := make([]float32, cfg.hSize/4)
					scr := make([]float32, cfg.hSize)
					var upd bool
					var es float32
					DlyMatchedFilterCore(l.lane, xs, thr, 0.7, x, y, h, &upd, &es, l.accum, acc, scr)
					what := fmt.Sprintf("%+v buffer %d lane %+v xs=%d", cfg, bi, l, xs)
					if upd {
						t.Fatalf("%s: the C++ core updated the filter on quiet render", what)
					}
					if i, ok := dlyEqualBits(h, h0); !ok {
						t.Fatalf("%s: h[%d] changed", what, i)
					}
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
		for _, l := range lanes {
			y := dlySpeechLike(rng, cfg.hSize/32, 5000)
			h := make([]float32, cfg.hSize)
			acc := make([]float32, cfg.hSize/4)
			scr := make([]float32, cfg.hSize)
			var upd bool
			var es float32
			DlyMatchedFilterCore(l.lane, 7, thr, 0.7, x, y, h, &upd, &es, l.accum, acc, scr)
			if !upd {
				t.Fatalf("%+v lane %+v: no update just above the excitation limit", cfg, l)
			}
		}
	}
}

// TestDlyMatchedFilterQuietParity drives aec3.MatchedFilter and
// webrtc::MatchedFilter with a low-rate render buffer written sample by
// sample: loud noise, zeros, every sample at +-quiet limit, random
// magnitudes below it, every sample one float above it, and constant
// magnitude 0.1% above the excitation limit (which updates the filters).
// The capture is the delayed render, so the filters lock whenever they run.
// The lag estimates must agree on every block, and the Go filter must take
// the quiet-render skip on some blocks and run on others.
func TestDlyMatchedFilterQuietParity(t *testing.T) {
	for _, factor := range []int{4, 8} {
		for _, preEcho := range []bool{false, true} {
			for _, opt := range []aec3.Aec3Optimization{aec3.OptNone, aec3.OptNeon} {
				t.Run(fmt.Sprintf("ds%d_preecho%v_opt%d", factor, preEcho, opt), func(t *testing.T) {
					rng := rand.New(rand.NewSource(int64(50 + factor)))
					sub := aec3.BlockSize / factor
					limit := float32(150)
					if factor == 8 {
						limit = 20
					}
					quiet, ok := aec3.MatchedFilterQuietLimit(32*sub, limit)
					if !ok {
						t.Fatal("no quiet limit")
					}
					buf := aec3.NewDownsampledRenderBuffer(aec3.GetDownSampledBufferSize(factor, 5))
					buf.Read = buf.OffsetIndex(buf.Write, sub)
					g := aec3.NewMatchedFilter(opt, sub, 32, 5, 24, limit, 0.7, 0.3, 0.2, preEcho)
					c := NewDlyMatchedFilter(opt, sub, 32, 5, 24, limit, 0.7, 0.3, 0.2, preEcho, len(buf.Buffer))
					defer c.Free()
					var history []float32
					lag := 150 + rng.Intn(100)
					sign := func() float32 {
						if rng.Intn(2) == 0 {
							return -1
						}
						return 1
					}
					const numBlocks = 6000
					skipsBefore := aec3.MatchedFilterQuietSkipsForTesting()
					reports := [6]int{}
					mode, left := 0, 0
					for blk := 0; blk < numBlocks; blk++ {
						if left == 0 {
							mode = rng.Intn(6)
							left = 20 + rng.Intn(3*len(buf.Buffer)/sub)
						}
						left--
						x := make([]float32, sub)
						for i := range x {
							switch mode {
							case 0:
								x[i] = float32(rng.NormFloat64() * 3000)
							case 1:
								x[i] = 0
							case 2:
								x[i] = sign() * quiet
							case 3:
								x[i] = sign() * quiet * float32(rng.Float64())
							case 4:
								x[i] = sign() * math.Nextafter32(quiet, float32(math.Inf(1)))
							case 5:
								x[i] = sign() * limit * 1.001
							}
						}
						history = append(history, x...)
						buf.UpdateWriteIndex(-sub)
						for i := range x {
							buf.Buffer[buf.Write+i] = x[sub-1-i]
						}
						buf.UpdateReadIndex(-sub)
						y := make([]float32, sub)
						for i := range y {
							if k := len(history) - sub - lag + i; k >= 0 {
								y[i] = 0.6 * history[k]
							}
							y[i] += float32(rng.NormFloat64())
						}
						slow := blk%7 == 0
						g.Update(buf, y, slow)
						eg, okg := g.GetBestLagEstimate()
						ec, okc := c.Update(buf, y, slow)
						if okg != okc || eg != ec {
							t.Fatalf("block %d (mode %d): got %v/%v want %v/%v", blk, mode, eg, okg, ec, okc)
						}
						if okc {
							reports[mode]++
						}
						if blk%997 == 500 {
							g.Reset(blk%2 == 0)
							c.Reset(blk%2 == 0)
						}
					}
					skips := int(aec3.MatchedFilterQuietSkipsForTesting() - skipsBefore)
					t.Logf("skipped %d of %d blocks; lag reports per mode %v", skips, numBlocks, reports)
					if skips < 200 || numBlocks-skips < 200 {
						t.Errorf("skipped %d of %d blocks: the scenario must both skip and run the filters", skips, numBlocks)
					}
					if reports[0] == 0 || reports[5] == 0 {
						t.Errorf("lag reports per mode %v: loud render and render just above the excitation limit must report lags",
							reports)
					}
				})
			}
		}
	}
}
