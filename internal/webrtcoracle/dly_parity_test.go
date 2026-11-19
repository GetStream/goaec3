//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// dlyEchoScene produces a render signal with speech-like structure (AR
// coloured noise with an on/off envelope) and a capture signal holding the
// render delayed by delay samples, scaled, plus noise. Signals are per
// band/channel and blockwise ([block][band][channel][64] flattened).
type dlyEchoScene struct {
	bands, renderCh, captureCh int
	render, capture            [][]float32 // per block, flat [band][channel][64]
}

func dlySpeechLike(rng *rand.Rand, n int, amp float64) []float32 {
	out := make([]float32, n)
	var s1, s2 float64
	env := 1.0
	for i := range out {
		if i%800 == 0 {
			if rng.Float64() < 0.25 {
				env = 0.02
			} else {
				env = 0.3 + rng.Float64()
			}
		}
		s1 = 0.6*s1 + rng.NormFloat64()
		s2 = 0.5*s2 + s1
		out[i] = float32(amp * env * s2 / 4)
	}
	return out
}

// dlyMakeScene builds numBlocks blocks; delayAt(block) gives the echo delay
// in samples for that block.
func dlyMakeScene(rng *rand.Rand, numBlocks, bands, renderCh, captureCh int, delayAt func(int) int,
	gain, noise float64) *dlyEchoScene {
	n := numBlocks * aec3.BlockSize
	sc := &dlyEchoScene{bands: bands, renderCh: renderCh, captureCh: captureCh}
	ren := make([][][]float32, bands)
	for b := range ren {
		ren[b] = make([][]float32, renderCh)
		for c := range ren[b] {
			amp := 6000.0
			if b > 0 {
				amp = 1500
			}
			if c > 0 {
				amp *= 0.5
			}
			ren[b][c] = dlySpeechLike(rng, n, amp)
		}
	}
	for blk := 0; blk < numBlocks; blk++ {
		r := make([]float32, 0, bands*renderCh*aec3.BlockSize)
		cp := make([]float32, 0, bands*captureCh*aec3.BlockSize)
		d := delayAt(blk)
		for b := 0; b < bands; b++ {
			for c := 0; c < renderCh; c++ {
				r = append(r, ren[b][c][blk*aec3.BlockSize:(blk+1)*aec3.BlockSize]...)
			}
			for c := 0; c < captureCh; c++ {
				for i := 0; i < aec3.BlockSize; i++ {
					idx := blk*aec3.BlockSize + i - d
					var v float64
					if idx >= 0 {
						v = gain * float64(ren[b][c%renderCh][idx])
					}
					v += noise * rng.NormFloat64()
					cp = append(cp, float32(v))
				}
			}
		}
		sc.render = append(sc.render, r)
		sc.capture = append(sc.capture, cp)
	}
	return sc
}

// dlyLockTracker records the C++ delay estimates of two phases of a scene
// whose true echo delay shifts between them, so the parity tests also prove
// that the estimator really tracks the echo.
type dlyLockTracker struct{ hist [2]map[int]int }

func newDlyLockTracker() *dlyLockTracker {
	return &dlyLockTracker{hist: [2]map[int]int{{}, {}}}
}

func (l *dlyLockTracker) add(phase, v int) { l.hist[phase][v]++ }

func (l *dlyLockTracker) mode(phase int) (int, int) {
	best, n := 0, -1
	for v, c := range l.hist[phase] {
		if c > n || (c == n && v < best) {
			best, n = v, c
		}
	}
	return best, n
}

// check requires both phases to hold at least minCount estimates, the first
// phase mode to lie at or somewhat below trueDelay (the reported delay
// excludes buffer latency and headroom), and the mode to move by wantShift.
func (l *dlyLockTracker) check(t *testing.T, trueDelay, wantShift, tol, minCount int) {
	t.Helper()
	m0, n0 := l.mode(0)
	m1, n1 := l.mode(1)
	if n0 < minCount || n1 < minCount {
		t.Fatalf("C++ reference produced too few estimates: %d (mode %d) and %d (mode %d)", n0, m0, n1, m1)
	}
	if m0 > trueDelay+tol || m0 < trueDelay/2-tol {
		t.Fatalf("C++ reference locked at %d, true delay %d", m0, trueDelay)
	}
	if d := m1 - m0 - wantShift; d > tol || d < -tol {
		t.Fatalf("C++ reference moved from %d to %d, want a shift of %d", m0, m1, wantShift)
	}
}

func dlyBlock(data []float32, bands, channels int) aec3.Block {
	b := aec3.NewBlock(bands, channels)
	for band := 0; band < bands; band++ {
		for ch := 0; ch < channels; ch++ {
			o := (band*channels + ch) * aec3.BlockSize
			copy(b.View(band, ch), data[o:o+aec3.BlockSize])
		}
	}
	return b
}

func dlyEqualBits(a, b []float32) (int, bool) {
	if len(a) != len(b) {
		return -1, false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i, false
		}
	}
	return 0, true
}

func TestDlyMatchedFilterCoreParity(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	type tc struct {
		lane    DlyMatchedFilterCoreLane
		accum   bool
		goCore  func(xs int, thr, sm float32, x, y, h []float32, upd *bool, es *float32, acc, scratch []float32)
		hSizes  []int
		minStep int
	}
	cases := []tc{
		{DlyCoreScalar, false, func(xs int, thr, sm float32, x, y, h []float32, upd *bool, es *float32, acc, _ []float32) {
			aec3.MatchedFilterCore(xs, thr, sm, x, y, h, upd, es, false, acc)
		}, []int{512, 256, 37, 4}, 1},
		{DlyCoreScalar, true, func(xs int, thr, sm float32, x, y, h []float32, upd *bool, es *float32, acc, _ []float32) {
			aec3.MatchedFilterCore(xs, thr, sm, x, y, h, upd, es, true, acc)
		}, []int{512, 256, 36}, 4},
		{DlyCoreNeon, false, func(xs int, thr, sm float32, x, y, h []float32, upd *bool, es *float32, acc, scr []float32) {
			aec3.MatchedFilterCoreNeon(xs, thr, sm, x, y, h, upd, es, false, acc, scr)
		}, []int{512, 256, 36, 8}, 1},
		{DlyCoreNeon, true, func(xs int, thr, sm float32, x, y, h []float32, upd *bool, es *float32, acc, scr []float32) {
			aec3.MatchedFilterCoreNeon(xs, thr, sm, x, y, h, upd, es, true, acc, scr)
		}, []int{512, 256, 36}, 4},
		{DlyCoreNeonAccumulatedError, true, func(xs int, thr, sm float32, x, y, h []float32, upd *bool, es *float32, acc, scr []float32) {
			aec3.MatchedFilterCoreWithAccumulatedErrorNeon(xs, thr, sm, x, y, h, upd, es, acc, scr)
		}, []int{512, 256, 36}, 4},
	}
	for ci, c := range cases {
		for _, hSize := range c.hSizes {
			for _, xSize := range []int{hSize + 3, 2*hSize + 1, 1012, 2000} {
				if xSize <= hSize {
					continue
				}
				x := dlySpeechLike(rng, xSize, 5000)
				hGo := make([]float32, hSize)
				hC := make([]float32, hSize)
				accGo := make([]float32, hSize/4)
				accC := make([]float32, hSize/4)
				scrGo := make([]float32, hSize+16) // the Go cores take the windows of a whole call
				scrC := make([]float32, hSize)
				thr := float32(hSize) * 150 * 150
				for call := 0; call < 60; call++ {
					y := dlySpeechLike(rng, 16, 5000)
					if call%17 == 3 {
						y[5] = 32000
					}
					// Start indices near the wrap exercise uneven chunk sizes.
					xs := rng.Intn(xSize)
					if call%3 == 0 {
						xs = xSize - 1 - rng.Intn(min(hSize, 20))
					}
					sm := float32(0.7)
					var updGo, updC bool
					var esGo, esC float32
					c.goCore(xs, thr, sm, x, y, hGo, &updGo, &esGo, accGo, scrGo)
					DlyMatchedFilterCore(c.lane, xs, thr, sm, x, y, hC, &updC, &esC, c.accum, accC, scrC)
					what := fmt.Sprintf("case %d h=%d x=%d call %d", ci, hSize, xSize, call)
					if i, ok := dlyEqualBits(hGo, hC); !ok {
						t.Fatalf("%s: h[%d] got %v want %v", what, i, hGo[i], hC[i])
					}
					if math.Float32bits(esGo) != math.Float32bits(esC) || updGo != updC {
						t.Fatalf("%s: errorSum/updated got %v/%v want %v/%v", what, esGo, updGo, esC, updC)
					}
					if c.accum {
						if i, ok := dlyEqualBits(accGo, accC); !ok {
							t.Fatalf("%s: accumulated[%d] got %v want %v", what, i, accGo[i], accC[i])
						}
					}
				}
			}
		}
	}
}

func TestDlyDecimatorParity(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, factor := range []int{4, 8} {
		g := aec3.NewDecimator(factor)
		c := NewDlyDecimator(factor)
		sig := dlySpeechLike(rng, 64*500, 8000)
		outG := make([]float32, aec3.BlockSize/factor)
		outC := make([]float32, aec3.BlockSize/factor)
		for blk := 0; blk < 500; blk++ {
			in := sig[blk*64 : (blk+1)*64]
			g.Decimate(in, outG)
			c.Decimate(in, outC)
			if i, ok := dlyEqualBits(outG, outC); !ok {
				t.Fatalf("factor %d block %d [%d]: got %v want %v", factor, blk, i, outG[i], outC[i])
			}
		}
		c.Free()
	}
}

// dlyGoLowRate fills a Go DownsampledRenderBuffer like RenderDelayBufferImpl:
// the write index moves backwards one sub-block per block and the decimated
// samples are stored reversed; read trails one sub-block behind.
type dlyGoLowRate struct {
	buf *aec3.DownsampledRenderBuffer
	dec *aec3.Decimator
	sub int
	ds  []float32
}

func newDlyGoLowRate(factor, numFilters int) *dlyGoLowRate {
	l := &dlyGoLowRate{buf: aec3.NewDownsampledRenderBuffer(aec3.GetDownSampledBufferSize(factor, numFilters)),
		dec: aec3.NewDecimator(factor), sub: aec3.BlockSize / factor}
	l.ds = make([]float32, l.sub)
	l.buf.Read = l.buf.OffsetIndex(l.buf.Write, l.sub)
	return l
}

func (l *dlyGoLowRate) push(x []float32) {
	l.buf.UpdateWriteIndex(-l.sub)
	l.dec.Decimate(x, l.ds)
	for i := range l.ds {
		l.buf.Buffer[l.buf.Write+i] = l.ds[len(l.ds)-1-i]
	}
	l.buf.UpdateReadIndex(-l.sub)
}

func TestDlyMatchedFilterParity(t *testing.T) {
	for _, factor := range []int{4, 8} {
		for _, preEcho := range []bool{false, true} {
			for _, opt := range []aec3.Aec3Optimization{aec3.OptNone, aec3.OptNeon} {
				name := fmt.Sprintf("ds%d_preecho%v_opt%d", factor, preEcho, opt)
				t.Run(name, func(t *testing.T) {
					rng := rand.New(rand.NewSource(12))
					const numBlocks = 1500
					trueDelay := 37 * factor * 4
					sc := dlyMakeScene(rng, numBlocks, 1, 1, 1, func(b int) int {
						if b > 900 {
							return trueDelay + 200
						}
						return trueDelay
					}, 0.6, 30)
					sub := aec3.BlockSize / factor
					lr := newDlyGoLowRate(factor, 5)
					capDec := aec3.NewDecimator(factor)
					limit := float32(150)
					if factor == 8 {
						limit = 20
					}
					g := aec3.NewMatchedFilter(opt, sub, 32, 5, 24, limit, 0.7, 0.3, 0.2, preEcho)
					c := NewDlyMatchedFilter(opt, sub, 32, 5, 24, limit, 0.7, 0.3, 0.2, preEcho, len(lr.buf.Buffer))
					defer c.Free()
					if g.GetMaxFilterLag() != c.GetMaxFilterLag() {
						t.Fatalf("max filter lag %d want %d", g.GetMaxFilterLag(), c.GetMaxFilterLag())
					}
					capDs := make([]float32, sub)
					lock := newDlyLockTracker()
					for blk := 0; blk < numBlocks; blk++ {
						lr.push(sc.render[blk])
						capDec.Decimate(sc.capture[blk], capDs)
						slow := blk > 400
						g.Update(lr.buf, capDs, slow)
						eg, okg := g.GetBestLagEstimate()
						ec, okc := c.Update(lr.buf, capDs, slow)
						if okg != okc || eg != ec {
							t.Fatalf("block %d: got %v/%v want %v/%v", blk, eg, okg, ec, okc)
						}
						if okc && blk < 900 {
							lock.add(0, ec.Lag*factor)
						} else if okc && blk > 1000 {
							lock.add(1, ec.Lag*factor)
						}
						if blk == 700 {
							g.Reset(false)
							c.Reset(false)
						}
					}
					lock.check(t, trueDelay, 200, 2*factor, 50)
				})
			}
		}
	}
}

func dlyTestConfigs() map[string]aec3.EchoCanceller3Config {
	out := map[string]aec3.EchoCanceller3Config{}
	def := aec3.DefaultEchoCanceller3Config()
	out["default"] = def
	c := def
	c.Delay.DetectPreEcho = false
	out["nopreecho"] = c
	c = def
	c.Delay.DownSamplingFactor = 8
	c.Delay.NumFilters = 4
	out["ds8"] = c
	c = def
	c.Delay.CaptureAlignmentMixing = aec3.AlignmentMixing{Downmix: true}
	c.Delay.RenderAlignmentMixing = aec3.AlignmentMixing{Downmix: true}
	c.Delay.DelayHeadroomSamples = 64
	c.Delay.HysteresisLimitBlocks = 3
	out["downmix"] = c
	c = def
	c.RenderLevels.RenderPowerGainDb = 4.5
	c.Delay.DelaySelectionThresholds.Initial = 3
	c.Delay.DelaySelectionThresholds.Converged = 10
	c.Delay.DelayEstimateSmoothingDelayFound = 0.5
	c.Filter.Refined.LengthBlocks = 20
	out["tuned"] = c
	return out
}

func TestDlyEchoPathDelayEstimatorParity(t *testing.T) {
	for name, cfg := range dlyTestConfigs() {
		for _, ch := range []int{1, 2} {
			t.Run(fmt.Sprintf("%s_ch%d", name, ch), func(t *testing.T) {
				cfg := cfg
				rng := rand.New(rand.NewSource(13))
				const numBlocks = 2500
				const trueDelay = 600
				sc := dlyMakeScene(rng, numBlocks, 1, 1, ch, func(b int) int {
					if b > 1500 {
						return trueDelay - 260
					}
					return trueDelay
				}, 0.5, 20)
				lr := newDlyGoLowRate(cfg.Delay.DownSamplingFactor, cfg.Delay.NumFilters)
				g := aec3.NewEchoPathDelayEstimator(&cfg, ch)
				c := NewDlyEchoPathDelayEstimator(&cfg, ch, len(lr.buf.Buffer))
				defer c.Free()
				lock := newDlyLockTracker()
				for blk := 0; blk < numBlocks; blk++ {
					lr.push(sc.render[blk])
					capture := dlyBlock(sc.capture[blk], 1, ch)
					dg, okg := g.EstimateDelay(lr.buf, &capture)
					dc, okc, cdc := c.EstimateDelay(lr.buf, sc.capture[blk], 1, ch)
					if okg != okc || dg != dc || int(g.Clockdrift()) != cdc {
						t.Fatalf("block %d: got %+v/%v/%d want %+v/%v/%d", blk, dg, okg, g.Clockdrift(), dc, okc, cdc)
					}
					if okc && blk < 1500 {
						lock.add(0, dc.Delay)
					} else if okc && blk > 1700 {
						lock.add(1, dc.Delay)
					}
					if blk == 1200 {
						g.Reset(true)
						c.Reset(true)
					}
				}
				lock.check(t, trueDelay, -260, 4*cfg.Delay.DownSamplingFactor, 100)
			})
		}
	}
}

// dlyCompareRdb compares the full RenderDelayBuffer state. withFft adds the
// FFT and spectrum rings, which in the default lane depend on the NEON Ooura
// FFT.
func dlyCompareRdb(t *testing.T, what string, g *aec3.RenderDelayBuffer, c *DlyRenderDelayBuffer, withFft bool) {
	t.Helper()
	s := c.State()
	rb := g.GetRenderBuffer()
	bb := rb.GetBlockBuffer()
	sb := rb.GetSpectrumBuffer()
	fb := rb.GetFftBuffer()
	lr := g.GetDownsampledRenderBuffer()
	if bb.Size != s.Size || len(lr.Buffer) != s.LowRateSize {
		t.Fatalf("%s: sizes %d/%d want %d/%d", what, bb.Size, len(lr.Buffer), s.Size, s.LowRateSize)
	}
	act := 0
	if rb.GetRenderActivity() {
		act = 1
	}
	idx := [9]int{bb.Read, bb.Write, sb.Read, sb.Write, rb.Position(), rb.Headroom(), lr.Read, lr.Write, act}
	if idx != s.Idx {
		t.Fatalf("%s: indices %v want %v", what, idx, s.Idx)
	}
	var blocks, spectra, re, im []float32
	for i := 0; i < bb.Size; i++ {
		blk := bb.At(i)
		for band := 0; band < blk.NumBands(); band++ {
			for ch := 0; ch < blk.NumChannels(); ch++ {
				blocks = append(blocks, blk.View(band, ch)...)
			}
		}
		for ch := range sb.At(i) {
			spectra = append(spectra, sb.At(i)[ch][:]...)
			re = append(re, fb.At(i)[ch].Re[:]...)
			im = append(im, fb.At(i)[ch].Im[:]...)
		}
	}
	type pair struct {
		n    string
		a, b []float32
	}
	pairs := []pair{{"blocks", blocks, s.Blocks}, {"lowrate", lr.Buffer, s.LowRate}}
	if withFft {
		pairs = append(pairs, pair{"fft.re", re, s.FftRe}, pair{"fft.im", im, s.FftIm},
			pair{"spectra", spectra, s.Spectra})
	}
	for _, p := range pairs {
		if i, ok := dlyEqualBits(p.a, p.b); !ok {
			if i < 0 {
				t.Fatalf("%s: %s length %d want %d", what, p.n, len(p.a), len(p.b))
			}
			t.Fatalf("%s: %s[%d] got %v want %v", what, p.n, i, p.a[i], p.b[i])
		}
	}
	if g.Delay() != c.Delay() || g.MaxDelay() != c.MaxDelay() || g.HasReceivedBufferDelay() != c.HasReceivedBufferDelay() {
		t.Fatalf("%s: delay/max/has %d/%d/%v want %d/%d/%v", what, g.Delay(), g.MaxDelay(), g.HasReceivedBufferDelay(),
			c.Delay(), c.MaxDelay(), c.HasReceivedBufferDelay())
	}
}

// dlyApiPattern returns the number of render inserts before capture block k:
// mostly 1, with bursts, starvation and a long render-only stretch that
// triggers overruns and excess-render detection.
func dlyApiPattern(rng *rand.Rand, k int) int {
	switch {
	case k >= 300 && k < 360:
		return 0 // render starvation: underruns
	case k == 400:
		return 40 // burst
	case k >= 500 && k < 760:
		if k%2 == 0 {
			return 2 // sustained excess render
		}
		return 1
	case k == 900:
		return 300 // overrun
	}
	return []int{1, 1, 1, 1, 2, 0}[rng.Intn(6)]
}

// TestDlyRenderDelayBufferParity compares events, indices, delays, the block
// ring and the low-rate buffer.
func TestDlyRenderDelayBufferParity(t *testing.T) { dlyRunRdbParity(t, false) }

// TestDlyRenderDelayBufferFftParity additionally compares the FFT and
// spectrum rings every capture block.
func TestDlyRenderDelayBufferFftParity(t *testing.T) { dlyRunRdbParity(t, true) }

func dlyRunRdbParity(t *testing.T, withFft bool) {
	type variant struct {
		rate, ch int
		cfgName  string
		external bool
	}
	cfgs := dlyTestConfigs()
	var variants []variant
	for _, rate := range []int{16000, 48000} {
		for _, ch := range []int{1, 2} {
			for _, name := range []string{"default", "ds8", "downmix", "tuned"} {
				variants = append(variants, variant{rate, ch, name, false})
			}
		}
	}
	variants = append(variants, variant{16000, 1, "default", true}, variant{48000, 2, "tuned", true})
	for _, v := range variants {
		t.Run(fmt.Sprintf("%s_%d_ch%d_ext%v", v.cfgName, v.rate, v.ch, v.external), func(t *testing.T) {
			cfg := cfgs[v.cfgName]
			if v.external {
				cfg.Delay.UseExternalDelayEstimator = true
			}
			bands := aec3.NumBandsForRate(v.rate)
			rng := rand.New(rand.NewSource(14))
			const numCapture = 1200
			sc := dlyMakeScene(rng, 4*numCapture, bands, v.ch, 1, func(int) int { return 0 }, 0, 0)
			g := aec3.NewRenderDelayBuffer(&cfg, v.rate, v.ch)
			c := NewDlyRenderDelayBuffer(&cfg, v.rate, v.ch)
			defer c.Free()
			dlyCompareRdb(t, "initial", g, c, withFft)
			ri := 0
			captureCalls := 0
			events := map[aec3.BufferingEvent]int{}
			for k := 0; k < numCapture; k++ {
				for n := dlyApiPattern(rng, k); n > 0 && ri < len(sc.render); n-- {
					blk := dlyBlock(sc.render[ri], bands, v.ch)
					eg := g.Insert(&blk)
					ec := c.Insert(sc.render[ri], bands, v.ch)
					ri++
					events[ec]++
					if eg != ec {
						t.Fatalf("capture %d insert: event %d want %d", k, eg, ec)
					}
				}
				captureCalls++
				if k%50 == 7 {
					g.HandleSkippedCaptureProcessing()
					c.HandleSkippedCaptureProcessing()
				} else {
					eg := g.PrepareCaptureProcessing()
					ec := c.PrepareCaptureProcessing()
					events[ec]++
					if eg != ec {
						t.Fatalf("capture %d prepare: event %d want %d", k, eg, ec)
					}
				}
				switch {
				case v.external && k%40 == 5:
					ms := 20 + 4*(k%13)
					g.SetAudioBufferDelay(ms)
					c.SetAudioBufferDelay(ms)
					// Keep the applied delay inside the ring, as the C++
					// index helpers require.
					d := ri - captureCalls + ms/4 - cfg.Delay.DelayHeadroomSamples/aec3.BlockSize
					if d >= 0 && d < g.GetRenderBuffer().GetBlockBuffer().Size {
						g.AlignFromExternalDelay()
						c.AlignFromExternalDelay()
					}
				case !v.external && k%25 == 3:
					d := rng.Intn(g.MaxDelay() + 4)
					if g.AlignFromDelay(d) != c.AlignFromDelay(d) {
						t.Fatalf("capture %d AlignFromDelay(%d) mismatch", k, d)
					}
				case k == 600:
					g.SetAudioBufferDelay(48)
					c.SetAudioBufferDelay(48)
					g.Reset()
					c.Reset()
				}
				dlyCompareRdb(t, fmt.Sprintf("capture %d", k), g, c, withFft)
			}
			if events[aec3.BufferingEventRenderOverrun] == 0 || events[aec3.BufferingEventRenderUnderrun] == 0 {
				t.Fatalf("scenario did not exercise overrun and underrun: %v", events)
			}
		})
	}
}

func TestDlyRenderDelayControllerParity(t *testing.T) {
	cfgs := dlyTestConfigs()
	for _, name := range []string{"default", "nopreecho", "ds8", "downmix", "tuned"} {
		for _, rate := range []int{16000, 48000} {
			for _, ch := range []int{1, 2} {
				t.Run(fmt.Sprintf("%s_%d_ch%d", name, rate, ch), func(t *testing.T) {
					cfg := cfgs[name]
					bands := aec3.NumBandsForRate(rate)
					rng := rand.New(rand.NewSource(15))
					const numBlocks = 3000
					const trueDelay = 700
					sc := dlyMakeScene(rng, numBlocks, bands, ch, ch, func(b int) int {
						switch {
						case b > 2200:
							return trueDelay + 3*64
						case b > 1500:
							return trueDelay - 5*64
						}
						return trueDelay
					}, 0.5, 20)
					gb := aec3.NewRenderDelayBuffer(&cfg, rate, ch)
					cb := NewDlyRenderDelayBuffer(&cfg, rate, ch)
					defer cb.Free()
					gc := aec3.NewRenderDelayController(&cfg, rate, ch)
					cc := NewDlyRenderDelayController(&cfg, rate, ch)
					defer cc.Free()
					lock := newDlyLockTracker()
					for blk := 0; blk < numBlocks; blk++ {
						rb := dlyBlock(sc.render[blk], bands, ch)
						if eg, ec := gb.Insert(&rb), cb.Insert(sc.render[blk], bands, ch); eg != ec {
							t.Fatalf("block %d insert event %d want %d", blk, eg, ec)
						}
						if eg, ec := gb.PrepareCaptureProcessing(), cb.PrepareCaptureProcessing(); eg != ec {
							t.Fatalf("block %d prepare event %d want %d", blk, eg, ec)
						}
						capture := dlyBlock(sc.capture[blk], bands, ch)
						dg, okg := gc.GetDelay(gb.GetDownsampledRenderBuffer(), gb.Delay(), &capture)
						dc, okc := cc.GetDelay(cb, sc.capture[blk], bands, ch)
						if okg != okc || dg != dc || gc.HasClockdrift() != cc.HasClockdrift() {
							t.Fatalf("block %d: got %+v/%v want %+v/%v", blk, dg, okg, dc, okc)
						}
						if okc {
							if gb.AlignFromDelay(dg.Delay) != cb.AlignFromDelay(dc.Delay) {
								t.Fatalf("block %d AlignFromDelay mismatch", blk)
							}
							if blk > 300 && blk < 1500 {
								lock.add(0, dc.Delay)
							} else if blk > 1800 && blk < 2200 {
								lock.add(1, dc.Delay)
							}
						}
						if blk == 1000 {
							gc.Reset(false)
							cc.Reset(false)
						}
						if blk%100 == 0 {
							dlyCompareRdb(t, fmt.Sprintf("block %d", blk), gb, cb, false)
						}
					}
					lock.check(t, trueDelay/aec3.BlockSize, -5, 1, 100)
				})
			}
		}
	}
}

func TestDlyFrameBlockerBlockFramerParity(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	for _, bands := range []int{1, 2, 3} {
		for _, ch := range []int{1, 2} {
			gBlocker := aec3.NewFrameBlocker(bands, ch)
			cBlocker := NewDlyFrameBlocker(bands, ch)
			gFramer := aec3.NewBlockFramer(bands, ch)
			cFramer := NewDlyBlockFramer(bands, ch)
			// The Go side reads and writes subframe k%2 of a frame.
			frame := aec3.NewFrame(bands, ch)
			outFrame := aec3.NewFrame(bands, ch)
			flat := func(f aec3.Frame, subFrameIndex int) []float32 {
				var o []float32
				for b := 0; b < bands; b++ {
					for c := 0; c < ch; c++ {
						o = append(o, f.SubFrame(b, c, subFrameIndex)...)
					}
				}
				return o
			}
			flatBlock := func(b *aec3.Block) []float32 {
				var o []float32
				for band := 0; band < bands; band++ {
					for c := 0; c < ch; c++ {
						o = append(o, b.View(band, c)...)
					}
				}
				return o
			}
			gBlock := aec3.NewBlock(bands, ch)
			cBlock := make([]float32, bands*ch*aec3.BlockSize)
			cSub := make([]float32, bands*ch*aec3.SubFrameLength)
			for k := 0; k < 200; k++ {
				for i, x := 0, frame.Samples(); i < len(x); i++ {
					x[i] = float32(rng.NormFloat64() * 1000)
				}
				gBlocker.InsertSubFrameAndExtractBlock(frame, k%2, &gBlock)
				cBlocker.InsertSubFrameAndExtractBlock(flat(frame, k%2), cBlock)
				if i, ok := dlyEqualBits(flatBlock(&gBlock), cBlock); !ok {
					t.Fatalf("bands %d ch %d k %d: blocker[%d] mismatch", bands, ch, k, i)
				}
				gFramer.InsertBlockAndExtractSubFrame(&gBlock, outFrame, k%2)
				cFramer.InsertBlockAndExtractSubFrame(cBlock, cSub)
				if i, ok := dlyEqualBits(flat(outFrame, k%2), cSub); !ok {
					t.Fatalf("bands %d ch %d k %d: framer[%d] mismatch", bands, ch, k, i)
				}
				if gBlocker.IsBlockAvailable() != cBlocker.IsBlockAvailable() {
					t.Fatalf("availability mismatch at %d", k)
				}
				if gBlocker.IsBlockAvailable() {
					gBlocker.ExtractBlock(&gBlock)
					cBlocker.ExtractBlock(cBlock)
					if i, ok := dlyEqualBits(flatBlock(&gBlock), cBlock); !ok {
						t.Fatalf("bands %d ch %d k %d: extract[%d] mismatch", bands, ch, k, i)
					}
					gFramer.InsertBlock(&gBlock)
					cFramer.InsertBlock(cBlock)
				}
			}
			cBlocker.Free()
			cFramer.Free()
		}
	}
}

func TestDlyBlockDelayBufferParity(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, rate := range []int{16000, 32000, 48000} {
		for _, ch := range []int{1, 2} {
			for _, delay := range []int{0, 1, 7, 64, 160, 333} {
				bands := rate / 16000
				g := aec3.NewBlockDelayBuffer(ch, bands, 160, delay)
				c := NewDlyBlockDelayBuffer(ch, rate, delay)
				frame := aec3.NewFrame(bands, ch)
				flatC := make([]float32, bands*ch*160)
				for k := 0; k < 30; k++ {
					for b := 0; b < bands; b++ {
						for c := 0; c < ch; c++ {
							x := frame.View(b, c)
							for i := range x {
								v := float32(rng.NormFloat64() * 1000)
								x[i] = v
								flatC[(b*ch+c)*160+i] = v
							}
						}
					}
					g.DelaySignal(frame)
					c.DelaySignal(flatC)
					for b := 0; b < bands; b++ {
						for cc := 0; cc < ch; cc++ {
							if i, ok := dlyEqualBits(frame.View(b, cc), flatC[(b*ch+cc)*160:(b*ch+cc+1)*160]); !ok {
								t.Fatalf("rate %d ch %d delay %d frame %d band %d ch %d [%d] mismatch", rate, ch, delay, k, b, cc, i)
							}
						}
					}
				}
				c.Free()
			}
		}
	}
}
