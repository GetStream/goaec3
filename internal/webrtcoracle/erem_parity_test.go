//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

const (
	eremRenderSize = 40
	eremReadLag    = 1
	eremMaxEcho    = 1024
)

// eremScenario describes one EchoRemover parity run.
type eremScenario struct {
	name                  string
	cfg                   aec3.EchoCanceller3Config
	rate                  int
	numRender, numCapture int
	blocks                int
	extDelay              bool // pass an external delay estimate
	linearAlways          bool // request the linear output on every block
	seed                  int64
}

// eremEnv generates multi-band render and capture signals and mirrors the
// render ring into a Go RenderBuffer and the C++ simulation.
type eremEnv struct {
	sc       *eremScenario
	rng      *rand.Rand
	numBands int
	bb       *aec3.BlockBuffer
	sb       *aec3.SpectrumBuffer
	fb       *aec3.FftBuffer
	rb       *aec3.RenderBuffer
	fft      aec3.Aec3Fft
	old      [][]float32
	hist     [][][]float64 // [band][render ch] history, newest last
	h        [][][]float64 // band-0 echo path [capture][render]
	hHigh    []float64     // upper-band echo path
	echoGain float64
	t        int
	capture  aec3.Block
}

func newEremEnv(sc *eremScenario) *eremEnv {
	nb := aec3.NumBandsForRate(sc.rate)
	e := &eremEnv{sc: sc, rng: rand.New(rand.NewSource(sc.seed)), numBands: nb,
		bb:       aec3.NewBlockBuffer(eremRenderSize, nb, sc.numRender),
		sb:       aec3.NewSpectrumBuffer(eremRenderSize, sc.numRender),
		fb:       aec3.NewFftBuffer(eremRenderSize, sc.numRender),
		echoGain: 1,
		capture:  aec3.NewBlock(nb, sc.numCapture),
	}
	e.rb = aec3.NewRenderBuffer(e.bb, e.sb, e.fb)
	e.old = make([][]float32, sc.numRender)
	for ch := range e.old {
		e.old[ch] = make([]float32, aec3.BlockSize)
	}
	e.hist = make([][][]float64, nb)
	for b := range e.hist {
		e.hist[b] = make([][]float64, sc.numRender)
		for ch := range e.hist[b] {
			e.hist[b][ch] = make([]float64, eremMaxEcho+aec3.BlockSize)
		}
	}
	e.newEchoPath(eremReadLag*aec3.BlockSize + 40)
	return e
}

// newEchoPath draws decaying random impulse responses with the given bulk
// delay in samples.
func (e *eremEnv) newEchoPath(delay int) {
	e.h = make([][][]float64, e.sc.numCapture)
	for c := range e.h {
		e.h[c] = make([][]float64, e.sc.numRender)
		for r := range e.h[c] {
			h := make([]float64, delay+200)
			for i := delay; i < len(h); i++ {
				h[i] = e.rng.NormFloat64() * 0.5 * math.Exp(-float64(i-delay)/30) / float64(e.sc.numRender)
			}
			h[delay] += 0.6 / float64(e.sc.numRender)
			e.h[c][r] = h
		}
	}
	e.hHigh = make([]float64, delay+40)
	for i := delay; i < len(e.hHigh); i++ {
		e.hHigh[i] = e.rng.NormFloat64() * 0.3 * math.Exp(-float64(i-delay)/8) / float64(e.sc.numRender)
	}
}

// nearend returns a speech-like burst sample (harmonics under a syllabic
// envelope) or 0 outside the bursts.
func (e *eremEnv) nearend(n, ch, band int) float64 {
	blk := e.t % 700
	if blk < 380 || blk >= 520 {
		return 0
	}
	env := math.Sin(math.Pi * float64(blk-380) / 140)
	env *= 0.5 + 0.5*math.Sin(2*math.Pi*float64(n)/2000)
	if band > 0 {
		return env * 600 * e.rng.NormFloat64()
	}
	f0 := 140 + 40*float64(ch) + 30*math.Sin(2*math.Pi*float64(n)/9000)
	var v float64
	for k := 1; k <= 8; k++ {
		v += math.Sin(2*math.Pi*f0*float64(k)*float64(n)/16000) / float64(k)
	}
	return env * 4000 * v
}

// step generates the next block, writes the render ring and the capture
// (Go block and C++ ring slot). It returns whether the capture saturates.
func (e *eremEnv) step(sim *EremSim, saturate bool) bool {
	e.t++
	pos := (e.fb.Write - 1 + eremRenderSize) % eremRenderSize
	seg := (e.t / 170) % 6
	amp := []float64{3000, 9000, 100, 6000, 15000, 0}[seg]
	for b := 0; b < e.numBands; b++ {
		for ch := 0; ch < e.sc.numRender; ch++ {
			hs := e.hist[b][ch]
			copy(hs, hs[aec3.BlockSize:])
			x := e.bb.At(pos).View(b, ch)
			for i := range x {
				n := e.t*aec3.BlockSize + i
				mod := 0.6 + 0.4*math.Sin(2*math.Pi*float64(n)/(3000+500*float64(ch)))
				a := amp
				if b > 0 {
					a *= 0.3
				}
				v := a * mod * e.rng.NormFloat64()
				if seg == 3 && b == 0 {
					v += 4000 * math.Sin(2*math.Pi*float64(n)*13/128)
				}
				v = math.Max(-32768, math.Min(32767, v))
				x[i] = float32(v)
				hs[len(hs)-aec3.BlockSize+i] = float64(x[i])
			}
		}
	}
	for ch := 0; ch < e.sc.numRender; ch++ {
		x := e.bb.At(pos).View(0, ch)
		X := &e.fb.At(pos)[ch]
		e.fft.PaddedFft(x, e.old[ch], aec3.WindowRectangular, X)
		copy(e.old[ch], x)
		X.Spectrum(aec3.OptNone, e.sb.At(pos)[ch][:])
	}
	blk := e.bb.At(pos)
	sim.SetRender(pos, &blk, e.fb.At(pos), e.sb.At(pos))
	read := (pos + eremReadLag) % eremRenderSize
	e.fb.Write, e.sb.Write, e.bb.Write = pos, pos, pos
	e.fb.Read, e.sb.Read, e.bb.Read = read, read, read
	sim.SetIndices(pos, read)

	saturated := false
	for b := 0; b < e.numBands; b++ {
		for c := 0; c < e.sc.numCapture; c++ {
			y := e.capture.View(b, c)
			for i := range y {
				n := e.t*aec3.BlockSize + i
				var v float64
				for r := 0; r < e.sc.numRender; r++ {
					hs := e.hist[b][r]
					idx := len(hs) - aec3.BlockSize + i
					h := e.hHigh
					if b == 0 {
						h = e.h[c][r]
					}
					for j, hj := range h {
						if hj != 0 {
							v += hj * hs[idx-j]
						}
					}
				}
				v = v*e.echoGain + e.nearend(n, c, b) + e.rng.NormFloat64()*20
				if saturate {
					v *= 8
				}
				if v >= 32767 || v <= -32768 {
					saturated = true
				}
				v = math.Max(-32768, math.Min(32767, v))
				y[i] = float32(v)
			}
		}
	}
	return saturated
}

func eremCmp(t *testing.T, blk int, what string, got, want []float32) {
	t.Helper()
	for k := range got {
		if math.Float32bits(got[k]) != math.Float32bits(want[k]) {
			t.Fatalf("block %d %s[%d]: go %v c++ %v", blk, what, k, got[k], want[k])
		}
	}
}

// eremRun drives a Go and a C++ EchoRemover identically and compares the
// outputs every block. kind and neon pick the C++ build and lane; opt is
// the Go lane (nil: NewEchoRemover, i.e. DetectOptimization()).
func eremRun(t *testing.T, sc eremScenario, kind EremKind, neon bool, opt *aec3.Aec3Optimization) {
	cfg := sc.cfg
	sim := NewEremSim(&cfg, sc.rate, sc.numRender, sc.numCapture, eremRenderSize, kind, neon)
	defer sim.Free()
	env := newEremEnv(&sc)

	var er *aec3.EchoRemover
	if opt == nil {
		er = aec3.NewEchoRemover(&cfg, sc.rate, sc.numRender, sc.numCapture)
	} else {
		er = aec3.NewEchoRemoverWithOptimization(&cfg, sc.rate, sc.numRender, sc.numCapture, *opt)
	}
	nb := aec3.NumBandsForRate(sc.rate)
	want := aec3.NewBlock(nb, sc.numCapture)
	linGo := aec3.NewBlock(1, sc.numCapture)
	linC := aec3.NewBlock(1, sc.numCapture)

	var nonzeroGain int
	for blk := 1; blk <= sc.blocks; blk++ {
		var v aec3.EchoPathVariability
		switch blk {
		case 700, 701, 703, 1100:
			// Consecutive gain changes exercise the gain-change hangover.
			env.echoGain = []float64{0.3, 0.5, 1.5, 1}[(blk/100)%4]
			v.GainChange = true
		case 1500:
			env.newEchoPath(eremReadLag*aec3.BlockSize + 150)
			v.DelayChange = aec3.DelayAdjustmentBufferFlush
		case 2300:
			env.newEchoPath(eremReadLag*aec3.BlockSize + 60)
			v.DelayChange = aec3.DelayAdjustmentNewDetectedDelay
			v.GainChange = true
		}
		v.ClockDrift = blk >= 1800 && blk < 1900
		saturate := blk >= 1950 && blk < 2010

		switch blk % 450 {
		case 100:
			er.UpdateEchoLeakageStatus(true)
			sim.UpdateEchoLeakageStatus(true)
		case 200:
			er.UpdateEchoLeakageStatus(false)
			sim.UpdateEchoLeakageStatus(false)
		}
		if blk == 1200 || blk == 2600 {
			er.SetCaptureOutputUsage(false)
			sim.SetCaptureOutputUsage(false)
		}
		if blk == 1280 || blk == 2650 {
			er.SetCaptureOutputUsage(true)
			sim.SetCaptureOutputUsage(true)
		}

		saturated := env.step(sim, saturate)
		var ext *aec3.DelayEstimate
		if sc.extDelay && blk > 50 {
			d := aec3.DelayEstimate{Quality: aec3.DelayQualityRefined, Delay: 3 + (blk/900)%3,
				BlocksSinceLastChange: blk % 900, BlocksSinceLastUpdate: blk % 7}
			ext = &d
		}
		withLinear := sc.linearAlways || (blk/250)%2 == 0
		var lg, lc *aec3.Block
		if withLinear {
			lg, lc = &linGo, &linC
		}

		want.CopyFrom(&env.capture)
		sim.ProcessCapture(v, saturated, ext, lc, &want)
		er.ProcessCapture(v, saturated, ext, env.rb, lg, &env.capture)

		for b := 0; b < nb; b++ {
			for ch := 0; ch < sc.numCapture; ch++ {
				eremCmp(t, blk, fmt.Sprintf("capture band %d ch %d", b, ch), env.capture.View(b, ch), want.View(b, ch))
			}
		}
		if withLinear {
			for ch := 0; ch < sc.numCapture; ch++ {
				eremCmp(t, blk, fmt.Sprintf("linear ch %d", ch), linGo.View(0, ch), linC.View(0, ch))
			}
		}
		m := er.GetMetrics()
		erl, erle := sim.GetMetrics()
		if math.Float64bits(m.EchoReturnLoss) != math.Float64bits(erl) ||
			math.Float64bits(m.EchoReturnLossEnhancement) != math.Float64bits(erle) {
			t.Fatalf("block %d metrics: go erl %v erle %v, c++ erl %v erle %v", blk, m.EchoReturnLoss,
				m.EchoReturnLossEnhancement, erl, erle)
		}
		if erle != 0 {
			nonzeroGain++
		}
	}
	if nonzeroGain == 0 {
		t.Errorf("ERLE metric never left zero")
	}
}

func eremScenarios() []eremScenario {
	def := aec3.DefaultEchoCanceller3Config()
	multi := aec3.CreateDefaultMultichannelConfig()

	exportLin := aec3.DefaultEchoCanceller3Config()
	exportLin.Filter.ExportLinearAecOutput = true

	noLinear := aec3.DefaultEchoCanceller3Config()
	noLinear.Filter.UseLinearFilter = false

	noCoarse := aec3.DefaultEchoCanceller3Config()
	noCoarse.Filter.EnableCoarseFilterOutputUsage = false

	conservative := aec3.DefaultEchoCanceller3Config()
	conservative.Filter.ConservativeInitialPhase = true
	conservative.Filter.InitialStateSeconds = 1.5
	conservative.EpStrength.BoundedErl = true
	conservative.EpStrength.EchoCanSaturate = false
	conservative.EpStrength.ErleOnsetCompensationInDominantNearend = true
	conservative.EchoAudibility.UseStationarityProperties = true
	conservative.EchoAudibility.UseStationarityPropertiesAtInit = true
	conservative.EchoRemovalControl.HasClockDrift = true
	conservative.Suppressor.ConservativeHfSuppression = true
	conservative.Suppressor.UseSubbandNearendDetection = true
	conservative.Suppressor.SubbandNearendDetection.Subband1 = aec3.SubbandRegion{Low: 1, High: 6}
	conservative.Suppressor.SubbandNearendDetection.Subband2 = aec3.SubbandRegion{Low: 7, High: 14}
	conservative.Suppressor.SubbandNearendDetection.NearendAverageBlocks = 2
	conservative.Suppressor.DominantNearendDetection.UseUnboundedEchoSpectrum = false
	conservative.Erle.NumSections = 3
	conservative.Erle.OnsetDetection = false

	stable := aec3.DefaultEchoCanceller3Config()
	stable.EchoRemovalControl.LinearAndStableEchoPath = true
	stable.Filter.Refined.LengthBlocks = 20
	stable.Filter.Coarse.LengthBlocks = 10
	stable.Filter.CoarseInitial.LengthBlocks = 8
	stable.EchoModel.ModelReverbInNonlinearMode = false
	stable.Suppressor.LfSmoothingDuringInitialPhase = false

	return []eremScenario{
		{name: "default_16k_1x1", cfg: def, rate: 16000, numRender: 1, numCapture: 1, blocks: 3200, seed: 1},
		{name: "default_32k_1x1_ext", cfg: def, rate: 32000, numRender: 1, numCapture: 1, blocks: 2800,
			extDelay: true, seed: 2},
		{name: "default_48k_2x2", cfg: multi, rate: 48000, numRender: 2, numCapture: 2, blocks: 2800, seed: 3},
		{name: "default_16k_2x1_ext", cfg: multi, rate: 16000, numRender: 2, numCapture: 1, blocks: 2700,
			extDelay: true, seed: 4},
		{name: "default_32k_1x2", cfg: def, rate: 32000, numRender: 1, numCapture: 2, blocks: 2700, seed: 5},
		{name: "export_linear_48k_1x1", cfg: exportLin, rate: 48000, numRender: 1, numCapture: 1, blocks: 2700,
			linearAlways: true, seed: 6},
		{name: "no_linear_filter_16k_1x2", cfg: noLinear, rate: 16000, numRender: 1, numCapture: 2, blocks: 2700,
			extDelay: true, seed: 7},
		{name: "no_coarse_output_32k_2x2", cfg: noCoarse, rate: 32000, numRender: 2, numCapture: 2, blocks: 2700,
			seed: 8},
		{name: "conservative_48k_1x1_ext", cfg: conservative, rate: 48000, numRender: 1, numCapture: 1,
			blocks: 2800, extDelay: true, seed: 9},
		{name: "stable_16k_1x1", cfg: stable, rate: 16000, numRender: 1, numCapture: 1, blocks: 2700, seed: 10},
	}
}

// TestEremEchoRemoverParity compares the Go EchoRemover with the C++ one.
// Default lane: the prebuilt EchoRemover::Create (kNeon) against
// NewEchoRemover, and the source build forced to kNone against OptNone.
// purego: the scalar-FFT source build in both lanes.
func TestEremEchoRemoverParity(t *testing.T) {
	none, neon := aec3.OptNone, aec3.OptNeon
	for _, sc := range eremScenarios() {
		if purego {
			t.Run(sc.name+"_none", func(t *testing.T) { eremRun(t, sc, EremScalarFft, false, nil) })
			t.Run(sc.name+"_neon", func(t *testing.T) { eremRun(t, sc, EremScalarFft, true, &neon) })
		} else {
			t.Run(sc.name+"_neon", func(t *testing.T) { eremRun(t, sc, EremPrebuilt, true, nil) })
			t.Run(sc.name+"_none", func(t *testing.T) { eremRun(t, sc, EremSource, false, &none) })
		}
	}
}

// TestEremLog10f checks that C++ log10f calls in the oracle binary resolve
// to the interposer (libm_interpose.c) and therefore agree with the Go log10f
// over a dense sweep of positive floats. The host log10f alone differs
// (e.g. at 1.1627032).
func TestEremLog10f(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	check := func(x float32) {
		got := float32(math.Log10(float64(x)))
		if want := EremLog10f(x); math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("log10f(%v): go %v host %v", x, got, want)
		}
	}
	for b := uint32(0x00800000); b < 0x7f800000; b += 97 {
		check(math.Float32frombits(b))
	}
	for i := 0; i < 1000000; i++ {
		check(float32(math.Exp(rng.Float64()*20 - 10)))
	}
}

func TestEremTransformDbMetricForReporting(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 200000; i++ {
		v := float32(math.Exp(rng.Float64()*40 - 20))
		if i%10 == 0 {
			v = 0
		}
		for _, p := range [][5]float32{{0, 59, 30, 1}, {0, 19, 0, 1}, {-5, 70, 3, 2.5}} {
			neg := i%2 == 0
			got := aec3.TransformDbMetricForReporting(neg, p[0], p[1], p[2], p[3], v)
			want := EremTransformDbMetricForReporting(neg, p[0], p[1], p[2], p[3], v)
			if got != want {
				t.Fatalf("TransformDbMetricForReporting(%v, %v, %v): go %d c++ %d", neg, p, v, got, want)
			}
		}
	}
}
