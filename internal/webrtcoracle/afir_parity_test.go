//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// afirEnv mirrors one render ring into a Go RenderBuffer and a C++
// RenderBuffer with identical contents.
type afirEnv struct {
	size, bands, channels int
	bb                    *aec3.BlockBuffer
	sb                    *aec3.SpectrumBuffer
	fb                    *aec3.FftBuffer
	rb                    *aec3.RenderBuffer
	c                     *AfirRenderBuffer
	fft                   aec3.Aec3Fft
	old                   [][]float32
	t                     int
	toneBin               float64
}

func newAfirEnv(size, bands, channels int) *afirEnv {
	e := &afirEnv{size: size, bands: bands, channels: channels,
		bb: aec3.NewBlockBuffer(size, bands, channels),
		sb: aec3.NewSpectrumBuffer(size, channels),
		fb: aec3.NewFftBuffer(size, channels),
		c:  NewAfirRenderBuffer(size, bands, channels)}
	e.rb = aec3.NewRenderBuffer(e.bb, e.sb, e.fb)
	e.old = make([][]float32, channels)
	for ch := range e.old {
		e.old[ch] = make([]float32, aec3.BlockSize)
	}
	return e
}

func (e *afirEnv) free() { e.c.Free() }

// push writes a new render block at the next read position (the ring moves
// backwards as in RenderDelayBuffer). Signals alternate between noise,
// loud tones (narrow band) and near-silence.
func (e *afirEnv) push(rng *rand.Rand) {
	pos := (e.fb.Read - 1 + e.size) % e.size
	e.t++
	phase := (e.t / 60) % 4
	if e.t%60 == 1 {
		e.toneBin = 3 + rng.Float64()*55
	}
	blk := &e.bb.Buffer[pos]
	for band := 0; band < e.bands; band++ {
		for ch := 0; ch < e.channels; ch++ {
			x := blk.View(band, ch)
			for i := range x {
				n := e.t*aec3.BlockSize + i
				var v float64
				switch phase {
				case 0:
					v = rng.NormFloat64() * 3000
				case 1:
					v = 8000*math.Sin(2*math.Pi*e.toneBin*float64(n)/128) + rng.NormFloat64()*2
				case 2:
					v = 5000*math.Sin(2*math.Pi*e.toneBin*float64(n)/128) + 3000*math.Sin(2*math.Pi*(e.toneBin/3+1)*float64(n)/128) + rng.NormFloat64()*200
				default:
					v = rng.NormFloat64() * 30
				}
				x[i] = float32(v)
			}
			e.c.SetBlock(pos, band, ch, x)
		}
	}
	for ch := 0; ch < e.channels; ch++ {
		x := blk.View(0, ch)
		X := &e.fb.Buffer[pos][ch]
		e.fft.PaddedFft(x, e.old[ch], aec3.WindowRectangular, X)
		copy(e.old[ch], x)
		X.Spectrum(aec3.OptNone, e.sb.Buffer[pos][ch][:])
		e.c.SetFft(pos, ch, X)
		e.c.SetSpectrum(pos, ch, &e.sb.Buffer[pos][ch])
	}
	e.fb.Read, e.sb.Read, e.bb.Read = pos, pos, pos
	e.c.SetRead(pos, pos, pos)
}

func afirCmpFft(t *testing.T, what string, got, want *aec3.FftData) {
	t.Helper()
	for k := range got.Re {
		if math.Float32bits(got.Re[k]) != math.Float32bits(want.Re[k]) || math.Float32bits(got.Im[k]) != math.Float32bits(want.Im[k]) {
			t.Fatalf("%s: bin %d go (%v,%v) c++ (%v,%v)", what, k, got.Re[k], got.Im[k], want.Re[k], want.Im[k])
		}
	}
}

func afirCmp(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len go %d c++ %d", what, len(got), len(want))
	}
	for k := range got {
		if math.Float32bits(got[k]) != math.Float32bits(want[k]) {
			t.Fatalf("%s: idx %d go %v c++ %v", what, k, got[k], want[k])
		}
	}
}

func afirRandH(rng *rand.Rand, P, C int, scale float64) [][]aec3.FftData {
	H := make([][]aec3.FftData, P)
	for p := range H {
		H[p] = make([]aec3.FftData, C)
		for ch := range H[p] {
			for k := range H[p][ch].Re {
				H[p][ch].Re[k] = float32(rng.NormFloat64() * scale)
				H[p][ch].Im[k] = float32(rng.NormFloat64() * scale)
			}
		}
	}
	return H
}

var afirLanes = []struct {
	name string
	opt  aec3.Aec3Optimization
	lane int
}{{"None", aec3.OptNone, AfirLaneScalar}, {"Neon", aec3.OptNeon, AfirLaneNeon}}

func TestAfirKernelsParity(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, numCh := range []int{1, 2, 3} {
		size := 17
		e := newAfirEnv(size, 1, numCh)
		for blk := 0; blk < 120; blk++ {
			e.push(rng)
			P := 1 + rng.Intn(size)
			np := rng.Intn(P + 1)
			H := afirRandH(rng, P, numCh, 0.05)
			flat := AfirFlatH(H)
			var G aec3.FftData
			for k := range G.Re {
				G.Re[k] = float32(rng.NormFloat64() * 1e-4)
				G.Im[k] = float32(rng.NormFloat64() * 1e-4)
			}
			for _, l := range afirLanes {
				name := fmt.Sprintf("ch%d blk%d %s", numCh, blk, l.name)
				var Sg, Sc aec3.FftData
				apply, adapt, fr, erl := aec3.ApplyFilter, aec3.AdaptPartitions, aec3.ComputeFrequencyResponse, aec3.ErlComputer
				if l.opt == aec3.OptNeon {
					apply, adapt, fr, erl = aec3.ApplyFilterNeon, aec3.AdaptPartitionsNeon, aec3.ComputeFrequencyResponseNeon, aec3.ErlComputerNEON
				}
				apply(e.rb, np, H, &Sg)
				AfirApplyFilter(e.c, l.lane, np, flat, P, numCh, &Sc)
				afirCmpFft(t, name+" ApplyFilter", &Sg, &Sc)

				Hg := make([][]aec3.FftData, P)
				for p := range H {
					Hg[p] = append([]aec3.FftData(nil), H[p]...)
				}
				fc := append([]aec3.FftData(nil), flat...)
				adapt(e.rb, &G, np, Hg)
				AfirAdaptPartitions(e.c, l.lane, &G, np, fc, P, numCh)
				fg := AfirFlatH(Hg)
				for i := range fg {
					afirCmpFft(t, name+" AdaptPartitions", &fg[i], &fc[i])
				}

				H2g := make([][aec3.FftLengthBy2Plus1]float32, P)
				H2c := make([][aec3.FftLengthBy2Plus1]float32, P)
				H2g[0][1], H2c[0][1] = 5, 5
				fr(np, Hg, H2g)
				AfirFreqResp(l.lane, np, fc, P, numCh, H2c)
				for p := range H2g {
					afirCmp(t, name+" FreqResp", H2g[p][:], H2c[p][:])
				}

				eg := make([]float32, aec3.FftLengthBy2Plus1)
				ec := make([]float32, aec3.FftLengthBy2Plus1)
				erl(H2g[:np], eg)
				AfirErl(l.lane, H2c[:np], ec)
				afirCmp(t, name+" Erl", eg, ec)
			}
		}
		e.free()
	}
}

// afirFftLaneScalar reports whether the Go Ooura FFT of this build is the
// scalar C path (purego) rather than the NEON path; the C++ reference
// filter is chosen to constrain through the same FFT.
const afirFftLaneScalar = purego

func TestAfirAdaptiveFirFilterParity(t *testing.T) {
	for _, l := range afirLanes {
		for _, numCh := range []int{1, 2} {
			t.Run(fmt.Sprintf("%s/ch%d", l.name, numCh), func(t *testing.T) {
				afirRunFilterParity(t, afirFftLaneScalar, l.opt, l.lane, numCh)
			})
		}
	}
}

func afirRunFilterParity(t *testing.T, scalarFft bool, opt aec3.Aec3Optimization, lane, numCh int) {
	rng := rand.New(rand.NewSource(int64(21 + lane*7 + numCh)))
	const maxSize, initSize, dur = 13, 10, 20
	size := maxSize + 7
	e := newAfirEnv(size, 1, numCh)
	defer e.free()
	g := aec3.NewAdaptiveFirFilter(maxSize, initSize, dur, numCh, opt)
	c := NewAfirFilter(scalarFft, maxSize, initSize, dur, numCh, lane)
	defer c.Free()
	Htrue := afirRandH(rng, maxSize, numCh, 0.3)
	irG := make([]float32, 0, aec3.GetTimeDomainLength(maxSize))
	irC := make([]float32, 0, aec3.GetTimeDomainLength(maxSize))
	H2g := make([][aec3.FftLengthBy2Plus1]float32, 0, maxSize)
	H2c := make([][aec3.FftLengthBy2Plus1]float32, 0, maxSize)
	hc := make([]aec3.FftData, maxSize*numCh)

	schedule := map[int]func(){
		40:  func() { g.SetSizePartitions(13, false); c.SetSizePartitions(13, false) },
		90:  func() { g.SetSizePartitions(4, false); c.SetSizePartitions(4, false) },
		100: func() { g.SetSizePartitions(12, false); c.SetSizePartitions(12, false) },
		150: func() { g.SetSizePartitions(2, true); c.SetSizePartitions(2, true) },
		160: func() { g.SetSizePartitions(13, true); c.SetSizePartitions(13, true) },
		170: func() { g.HandleEchoPathChange(); c.HandleEchoPathChange() },
		180: func() { g.SetSizePartitions(7, true); c.SetSizePartitions(7, true) },
		185: func() { g.HandleEchoPathChange(); c.HandleEchoPathChange() },
		190: func() { g.ScaleFilter(0.7); c.ScaleFilter(0.7) },
		200: func() { g.SetSizePartitions(1, false); c.SetSizePartitions(1, false) },
		230: func() { g.SetSizePartitions(13, false); c.SetSizePartitions(13, false) },
		240: func() {
			Hs := afirRandH(rng, maxSize, numCh, 0.01)
			g.SetFilter(9, Hs)
			c.SetFilter(9, AfirFlatH(Hs))
		},
		300: func() { g.SetSizePartitions(5, false); c.SetSizePartitions(5, false) },
	}

	for blk := 0; blk < 400; blk++ {
		e.push(rng)
		if f, ok := schedule[blk]; ok {
			f()
		}
		if g.SizePartitions() != c.SizePartitions() {
			t.Fatalf("blk %d: size go %d c++ %d", blk, g.SizePartitions(), c.SizePartitions())
		}
		var Sg, Sc aec3.FftData
		g.Filter(e.rb, &Sg)
		c.Filter(e.c, &Sc)
		afirCmpFft(t, fmt.Sprintf("blk %d Filter", blk), &Sg, &Sc)

		// Error against a synthetic echo path, NLMS-like gain.
		var Y, G aec3.FftData
		aec3.ApplyFilter(e.rb, maxSize, Htrue, &Y)
		var X2 [aec3.FftLengthBy2Plus1]float32
		e.rb.SpectralSum(g.SizePartitions(), &X2)
		for k := range G.Re {
			mu := float32(0.5) / (X2[k] + 1e6)
			G.Re[k] = mu * (Y.Re[k] - Sg.Re[k])
			G.Im[k] = mu * (Y.Im[k] - Sg.Im[k])
		}
		if blk%3 == 0 {
			g.Adapt(e.rb, &G)
			c.Adapt(e.c, &G)
		} else {
			g.AdaptWithImpulseResponse(e.rb, &G, &irG)
			irC = c.AdaptWithImpulseResponse(e.c, &G, irC)
			afirCmp(t, fmt.Sprintf("blk %d impulse response", blk), irG, irC)
		}
		g.ComputeFrequencyResponse(&H2g)
		H2c = c.ComputeFrequencyResponse(H2c)
		if len(H2g) != len(H2c) {
			t.Fatalf("blk %d: H2 len go %d c++ %d", blk, len(H2g), len(H2c))
		}
		for p := range H2g {
			afirCmp(t, fmt.Sprintf("blk %d H2[%d]", blk, p), H2g[p][:], H2c[p][:])
		}
		c.GetFilter(hc)
		hg := AfirFlatH(g.GetFilter())
		for i := range hg {
			afirCmpFft(t, fmt.Sprintf("blk %d H[%d]", blk, i), &hg[i], &hc[i])
		}
	}
}

func TestAfirRenderSignalAnalyzerParity(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for _, cfg := range []struct{ bands, ch, length int }{{1, 1, 13}, {2, 1, 13}, {3, 2, 5}} {
		e := newAfirEnv(20, cfg.bands, cfg.ch)
		conf := aec3.DefaultEchoCanceller3Config()
		conf.Filter.Refined.LengthBlocks = cfg.length
		g := aec3.NewRenderSignalAnalyzer(&conf)
		c := NewAfirRenderSignalAnalyzer(cfg.length)
		peaks, poor, masked := 0, 0, 0
		for blk := 0; blk < 1500; blk++ {
			e.push(rng)
			delay, has := rng.Intn(15), (blk/37)%5 != 0
			g.Update(e.rb, delay, has)
			c.Update(e.c, delay, has)
			if g.PoorSignalExcitation() != c.PoorSignalExcitation() {
				t.Fatalf("blk %d: PoorSignalExcitation go %v c++ %v", blk, g.PoorSignalExcitation(), c.PoorSignalExcitation())
			}
			bg, okg := g.NarrowPeakBand()
			bc, okc := c.NarrowPeakBand()
			if okg != okc || (okg && bg != bc) {
				t.Fatalf("blk %d: NarrowPeakBand go (%d,%v) c++ (%d,%v)", blk, bg, okg, bc, okc)
			}
			if okg {
				peaks++
			}
			if g.PoorSignalExcitation() {
				poor++
			}
			var vg [aec3.FftLengthBy2Plus1]float32
			for k := range vg {
				vg[k] = rng.Float32() + 0.5
			}
			vc := vg
			g.MaskRegionsAroundNarrowBands(&vg)
			c.MaskRegionsAroundNarrowBands(&vc)
			afirCmp(t, fmt.Sprintf("blk %d mask", blk), vg[:], vc[:])
			for _, v := range vg {
				if v == 0 {
					masked++
					break
				}
			}
		}
		if peaks == 0 || poor == 0 || masked == 0 {
			t.Fatalf("bands %d: peaks %d poor %d masked %d; inputs do not exercise the analyzer", cfg.bands, peaks, poor, masked)
		}
		c.Free()
		e.free()
	}
}

func afirRandSpectrumE(rng *rand.Rand, E *aec3.FftData, E2 *[aec3.FftLengthBy2Plus1]float32, scale float64) {
	for k := range E.Re {
		E.Re[k] = float32(rng.NormFloat64() * scale)
		E.Im[k] = float32(rng.NormFloat64() * scale)
	}
	E.Im[0], E.Im[aec3.FftLengthBy2] = 0, 0
	E.Spectrum(aec3.OptNone, E2[:])
}

func TestAfirFilterUpdateGainsParity(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	conf := aec3.DefaultEchoCanceller3Config()
	e := newAfirEnv(20, 1, 2)
	defer e.free()
	rg := aec3.NewRenderSignalAnalyzer(&conf)
	rc := NewAfirRenderSignalAnalyzer(conf.Filter.Refined.LengthBlocks)
	defer rc.Free()
	const dur = 30
	cg := aec3.NewCoarseFilterUpdateGain(conf.Filter.CoarseInitial, dur)
	cc := NewAfirCoarseGain(conf.Filter.CoarseInitial, dur)
	defer cc.Free()
	fg := aec3.NewRefinedFilterUpdateGain(conf.Filter.RefinedInitial, dur)
	fc := NewAfirRefinedGain(conf.Filter.RefinedInitial, dur)
	defer fc.Free()

	altCoarse := aec3.CoarseConfiguration{LengthBlocks: 10, Rate: 0.3, NoiseGate: 1e6}
	altRefined := aec3.RefinedConfiguration{LengthBlocks: 8, LeakageConverged: 0.001, LeakageDiverged: 0.2, ErrorFloor: 0.01, ErrorCeil: 5, NoiseGate: 1e6}
	updates, coarseUpdates := 0, 0
	for blk := 0; blk < 2000; blk++ {
		e.push(rng)
		rg.Update(e.rb, 3, blk%400 > 20)
		rc.Update(e.c, 3, blk%400 > 20)
		switch blk {
		case 100:
			cg.SetConfig(conf.Filter.Coarse, false)
			cc.SetConfig(conf.Filter.Coarse, false)
			fg.SetConfig(conf.Filter.Refined, false)
			fc.SetConfig(conf.Filter.Refined, false)
		case 300:
			cg.SetConfig(altCoarse, true)
			cc.SetConfig(altCoarse, true)
			fg.SetConfig(altRefined, true)
			fc.SetConfig(altRefined, true)
		case 500:
			cg.SetConfig(conf.Filter.Coarse, false)
			cc.SetConfig(conf.Filter.Coarse, false)
			fg.SetConfig(conf.Filter.Refined, false)
			fc.SetConfig(conf.Filter.Refined, false)
		case 515:
			cg.SetConfig(altCoarse, false)
			cc.SetConfig(altCoarse, false)
			fg.SetConfig(altRefined, false)
			fc.SetConfig(altRefined, false)
		}
		if blk%97 == 50 {
			v := aec3.EchoPathVariability{GainChange: rng.Intn(2) == 0, DelayChange: aec3.DelayAdjustment(rng.Intn(3)), ClockDrift: rng.Intn(2) == 0}
			cg.HandleEchoPathChange()
			cc.HandleEchoPathChange()
			fg.HandleEchoPathChange(v)
			fc.HandleEchoPathChange(v)
		}

		var X2 [aec3.FftLengthBy2Plus1]float32
		sizePartitions := 1 + rng.Intn(13)
		e.rb.SpectralSum(sizePartitions, &X2)
		var so aec3.SubtractorOutput
		scale := 3e4 * math.Pow(10, rng.Float64()*2-1)
		afirRandSpectrumE(rng, &so.ERefinedFft, &so.E2Refined, scale)
		var Ec aec3.FftData
		afirRandSpectrumE(rng, &Ec, &so.E2Coarse, scale*(0.5+rng.Float64()))
		erl := make([]float32, aec3.FftLengthBy2Plus1)
		for k := range erl {
			erl[k] = float32(rng.Float64() * 3)
		}
		saturated := rng.Intn(25) == 0
		disallow := rng.Intn(4) == 0

		var Gg, Gc aec3.FftData
		Gc.Re[2] = 9 // must be overwritten
		Gg.Re[2] = 9
		cg.Compute(&X2, rg, &Ec, sizePartitions, saturated, &Gg)
		cc.Compute(&X2, rc, &Ec, sizePartitions, saturated, &Gc)
		afirCmpFft(t, fmt.Sprintf("blk %d coarse G", blk), &Gg, &Gc)
		if Gg.Re[10] != 0 {
			coarseUpdates++
		}

		Gg, Gc = aec3.FftData{}, aec3.FftData{}
		fg.Compute(&X2, rg, &so, erl, sizePartitions, saturated, disallow, &Gg)
		fc.Compute(&X2, rc, &so, erl, sizePartitions, saturated, disallow, &Gc)
		afirCmpFft(t, fmt.Sprintf("blk %d refined G", blk), &Gg, &Gc)
		if Gg.Re[10] != 0 {
			updates++
		}
	}
	if updates < 200 || coarseUpdates < 200 {
		t.Fatalf("refined updates %d, coarse updates %d; inputs do not exercise the gains", updates, coarseUpdates)
	}
}
