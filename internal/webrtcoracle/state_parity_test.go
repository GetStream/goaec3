//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

const stK = aec3.FftLengthBy2Plus1

func stEqF(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: go %v (%08x) c++ %v (%08x)", what, i, got[i], math.Float32bits(got[i]),
				want[i], math.Float32bits(want[i]))
		}
	}
}

func stEqF1(t *testing.T, what string, got, want float32) {
	t.Helper()
	stEqF(t, what, []float32{got}, []float32{want})
}

func stEqSpectra(t *testing.T, what string, got, want [][stK]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d channels vs %d", what, len(got), len(want))
	}
	for ch := range want {
		stEqF(t, fmt.Sprintf("%s ch%d", what, ch), got[ch][:], want[ch][:])
	}
}

func stEqQ(t *testing.T, what string, got, want []aec3.LinearFilterQuality) {
	t.Helper()
	for i := range want {
		if got[i].Ok != want[i].Ok || (want[i].Ok && math.Float32bits(got[i].Value) != math.Float32bits(want[i].Value)) {
			t.Fatalf("%s[%d]: go %+v c++ %+v", what, i, got[i], want[i])
		}
	}
}

func stCheck(t *testing.T, what string, got, want any) {
	t.Helper()
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("%s: go %v c++ %v", what, got, want)
	}
}

// stRenderSim drives a paired Go/C++ render buffer with a slowly varying
// synthetic render signal. The write and read indices move backwards as in
// RenderDelayBuffer, with read lagging write by a fixed delay.
type stRenderSim struct {
	r     *StRender
	rng   *rand.Rand
	write int
	delay int
	t     int
	block []float32
	X2    [stK]float32
	// Level is the current render RMS level per channel.
	Level []float64
}

func newStRenderSim(rng *rand.Rand, size, numChannels, delay int) *stRenderSim {
	s := &stRenderSim{r: NewStRender(size, numChannels), rng: rng, delay: delay, block: make([]float32, aec3.BlockSize),
		Level: make([]float64, numChannels)}
	// Prefill every slot so that all reads see defined data.
	for i := 0; i < size; i++ {
		s.Advance()
	}
	return s
}

func (s *stRenderSim) envelope(ch int) float64 {
	t := float64(s.t)
	// Speech-like bursts with silent gaps and slow level drift.
	burst := 0.5 + 0.5*math.Sin(2*math.Pi*t/(310+37*float64(ch)))
	if math.Sin(2*math.Pi*t/900) < -0.6 {
		burst = 0.002
	}
	return 3000 * burst * (0.6 + 0.4*math.Sin(2*math.Pi*t/2700))
}

func (s *stRenderSim) Advance() {
	s.t++
	size := s.r.Size
	s.write = (s.write - 1 + size) % size
	for ch := 0; ch < s.r.Channels; ch++ {
		level := s.envelope(ch)
		if s.rng.Intn(400) == 0 {
			level = 0
		}
		s.Level[ch] = level
		for i := range s.block {
			s.block[i] = float32(level * s.rng.NormFloat64())
		}
		s.r.SetBlock(s.write, ch, s.block)
		for k := range s.X2 {
			shape := 1.0 / (1 + float64(k)/12)
			g := s.rng.NormFloat64()
			h := s.rng.NormFloat64()
			s.X2[k] = float32(128 * level * level * shape * 0.5 * (g*g + h*h))
		}
		s.r.SetSpectrum(s.write, ch, &s.X2)
	}
	s.r.SetIndices(s.write, (s.write+s.delay)%size)
}

func stRandSpectra(rng *rand.Rand, n int, scale float64) [][stK]float32 {
	out := make([][stK]float32, n)
	for ch := range out {
		for k := range out[ch] {
			g := rng.NormFloat64()
			out[ch][k] = float32(scale * (0.05 + g*g))
		}
	}
	return out
}

func stRandBools(rng *rand.Rand, n int, p float64) []bool {
	out := make([]bool, n)
	for i := range out {
		out[i] = rng.Float64() < p
	}
	return out
}

// stEchoPath models a converging adaptive filter towards a true echo path.
type stEchoPath struct {
	numCh, lenBlocks int
	peakBlock        int
	decay            float64
	gain             float64
	since            int
	trueH            [][]float32
	H                [][]float32
	H2               [][][stK]float32
}

func newStEchoPath(numCh, lenBlocks int) *stEchoPath {
	p := &stEchoPath{numCh: numCh, lenBlocks: lenBlocks}
	p.H = make([][]float32, numCh)
	p.trueH = make([][]float32, numCh)
	p.H2 = make([][][stK]float32, numCh)
	for ch := 0; ch < numCh; ch++ {
		p.H[ch] = make([]float32, lenBlocks*aec3.FftLengthBy2)
		p.trueH[ch] = make([]float32, lenBlocks*aec3.FftLengthBy2)
		p.H2[ch] = make([][stK]float32, lenBlocks)
	}
	return p
}

func (p *stEchoPath) Change(rng *rand.Rand, peakBlock int, decay, gain float64) {
	p.peakBlock, p.decay, p.gain, p.since = peakBlock, decay, gain, 0
	for ch := 0; ch < p.numCh; ch++ {
		h := p.trueH[ch]
		peak := peakBlock*aec3.FftLengthBy2 + 10 + 3*ch
		for i := range h {
			h[i] = 0
			if i >= peak {
				d := float64(i-peak) / aec3.FftLengthBy2
				h[i] = float32(gain * math.Pow(decay, d) * rng.NormFloat64() * 0.3)
			}
		}
		h[peak] = float32(gain)
	}
}

func (p *stEchoPath) Step(rng *rand.Rand) {
	p.since++
	conv := 1 - math.Exp(-float64(p.since)/400)
	noise := 0.02 * math.Exp(-float64(p.since)/600)
	for ch := 0; ch < p.numCh; ch++ {
		for i := range p.H[ch] {
			p.H[ch][i] = float32(conv*float64(p.trueH[ch][i]) + noise*rng.NormFloat64())
		}
		for b := 0; b < p.lenBlocks; b++ {
			var e float64
			for _, v := range p.H[ch][b*aec3.FftLengthBy2 : (b+1)*aec3.FftLengthBy2] {
				e += float64(v) * float64(v)
			}
			for k := 0; k < stK; k++ {
				p.H2[ch][b][k] = float32(e * (0.6 + 0.4*math.Cos(float64(k*(b+1))/9)) * (0.9 + 0.2*rng.Float64()))
			}
		}
	}
}

func stTestConfigs() map[string]aec3.EchoCanceller3Config {
	out := map[string]aec3.EchoCanceller3Config{}
	def := aec3.DefaultEchoCanceller3Config()
	out["default"] = def
	c := def
	c.EpStrength.DefaultLen = -0.83
	c.EpStrength.NearendLen = -0.5
	out["adaptive_decay"] = c
	c = def
	c.Erle.NumSections = 4
	out["signal_dependent_erle"] = c
	c = def
	c.EchoAudibility.UseStationarityProperties = true
	c.EchoAudibility.UseStationarityPropertiesAtInit = true
	out["stationarity"] = c
	c = def
	c.EchoAudibility.UseStationarityProperties = true
	c.Filter.ConservativeInitialPhase = true
	c.EpStrength.EchoCanSaturate = false
	out["conservative"] = c
	c = def
	c.EpStrength.BoundedErl = true
	c.EpStrength.UseConservativeTailFrequencyResponse = false
	c.EpStrength.ErleOnsetCompensationInDominantNearend = true
	c.Erle.OnsetDetection = false
	out["bounded_erl"] = c
	c = def
	c.Filter.UseLinearFilter = false
	c.EchoModel.ModelReverbInNonlinearMode = false
	out["no_linear_filter"] = c
	c = def
	c.Erle.NumSections = 3
	c.EpStrength.DefaultLen = -0.9
	c.Erle.ClampQualityEstimateToZero = false
	c.Erle.ClampQualityEstimateToOne = false
	c.EchoRemovalControl.LinearAndStableEchoPath = true
	c.Filter.Refined.LengthBlocks = 20
	c.Delay.DelayHeadroomSamples = 64
	out["long_filter"] = c
	return out
}

func TestStateMovingAverage(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, memLen := range []int{1, 2, 3, 6} {
		g := aec3.NewMovingAverage(stK, memLen)
		c := NewStMovingAverage(stK, memLen)
		in := make([]float32, stK)
		og, oc := make([]float32, stK), make([]float32, stK)
		for i := 0; i < 300; i++ {
			for k := range in {
				in[k] = float32(rng.NormFloat64() * 1e4)
			}
			g.Average(in, og)
			c.Average(in, oc)
			stEqF(t, "average", og, oc)
		}
		c.Free()
	}
}

func TestStateReverbModel(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	g := aec3.NewReverbModel()
	c := NewStReverbModel()
	defer c.Free()
	ps := make([]float32, stK)
	sc := make([]float32, stK)
	for i := 0; i < 2000; i++ {
		for k := range ps {
			ps[k] = float32(rng.ExpFloat64() * 1e7)
			sc[k] = float32(rng.Float64())
		}
		decay := float32(rng.Float64()*1.1 - 0.1)
		if i%2 == 0 {
			s := float32(rng.Float64())
			g.UpdateReverbNoFreqShaping(ps, s, decay)
			c.UpdateReverbNoFreqShaping(ps, s, decay)
		} else {
			g.UpdateReverb(ps, sc, decay)
			c.UpdateReverb(ps, sc, decay)
		}
		if i%500 == 499 {
			g.Reset()
			c.Free()
			c = NewStReverbModel()
		}
		want := c.Reverb()
		stEqF(t, "reverb", g.Reverb()[:], want[:])
	}
}

func TestStateReverbFrequencyResponse(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, conservative := range []bool{false, true} {
		g := aec3.NewReverbFrequencyResponse(conservative)
		c := NewStReverbFrequencyResponse(conservative)
		for i := 0; i < 2000; i++ {
			fr := stRandSpectra(rng, 13, 1e-2)
			if rng.Intn(50) == 0 {
				fr[3] = [stK]float32{}
			}
			delay := rng.Intn(13)
			if rng.Intn(50) == 0 {
				delay = 3
			}
			q := aec3.LinearFilterQuality{Value: float32(rng.Float64()), Ok: rng.Intn(5) != 0}
			stationary := rng.Intn(6) == 0
			g.Update(fr, delay, q, stationary)
			c.Update(fr, delay, q, stationary)
			want := c.FrequencyResponse()
			stEqF(t, "tail response", g.FrequencyResponse()[:], want[:])
		}
		c.Free()
	}
}

func TestStateReverbDecayEstimator(t *testing.T) {
	for name, cfg := range stTestConfigs() {
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(4))
			g := aec3.NewReverbDecayEstimator(&cfg)
			c := NewStReverbDecayEstimator(&cfg)
			defer c.Free()
			lb := cfg.Filter.Refined.LengthBlocks
			path := newStEchoPath(1, lb)
			path.Change(rng, 2, 0.6, 0.5)
			for i := 0; i < 20000; i++ {
				if i%3000 == 2999 {
					path.Change(rng, 1+rng.Intn(lb-4), 0.3+0.6*rng.Float64(), 0.1+rng.Float64())
				}
				path.Step(rng)
				filter := path.H[0]
				q := aec3.LinearFilterQuality{Value: float32(0.5 + 0.5*rng.Float64()), Ok: rng.Intn(4) != 0}
				delay := path.peakBlock
				if rng.Intn(200) == 0 {
					delay = 0
				}
				usable := rng.Intn(100) != 0
				stationary := rng.Intn(50) == 0
				if rng.Intn(1000) == 0 {
					filter = filter[:len(filter)-aec3.FftLengthBy2]
				}
				g.Update(filter, q, delay, usable, stationary)
				c.Update(filter, q, delay, usable, stationary)
				stEqF1(t, "decay", g.Decay(false), c.Decay(false))
				stEqF1(t, "mild decay", g.Decay(true), c.Decay(true))
			}
		})
	}
}

func TestStateErlEstimator(t *testing.T) {
	for _, nch := range [][2]int{{1, 1}, {2, 1}, {1, 2}, {2, 3}} {
		rng := rand.New(rand.NewSource(5))
		g := aec3.NewErlEstimator(10)
		c := NewStErlEstimator(10)
		for i := 0; i < 3000; i++ {
			X2 := stRandSpectra(rng, nch[0], math.Exp(rng.Float64()*4)*2e7)
			Y2 := stRandSpectra(rng, nch[1], math.Exp(rng.Float64()*4)*1e6)
			conv := stRandBools(rng, nch[1], 0.7)
			if i%700 == 699 {
				g.Reset()
				c.Reset()
			}
			g.Update(conv, X2, Y2)
			c.Update(conv, X2, Y2)
			erl, td := c.Get()
			stEqF(t, "erl", g.Erl()[:], erl[:])
			stEqF1(t, "erl td", g.ErlTimeDomain(), td)
		}
		c.Free()
	}
}

// stErleInputs synthesizes render, capture and residual spectra around the
// estimators' energy thresholds.
func stErleInputs(rng *rand.Rand, nch int, t int) (*[stK]float32, [][stK]float32, [][stK]float32) {
	var X2 [stK]float32
	level := 4.4e7 * math.Exp(3*math.Sin(float64(t)/80)+rng.NormFloat64()*0.3)
	for k := range X2 {
		g := rng.NormFloat64()
		X2[k] = float32(level * (0.1 + g*g) / (1 + float64(k)/30))
	}
	Y2 := make([][stK]float32, nch)
	E2 := make([][stK]float32, nch)
	erle := 2 + 20*(1-math.Exp(-float64(t)/500))
	for ch := 0; ch < nch; ch++ {
		for k := range Y2[ch] {
			Y2[ch][k] = float32(float64(X2[k]) * (0.3 + 0.2*rng.Float64()))
			if rng.Intn(30) == 0 {
				E2[ch][k] = 0
			} else {
				E2[ch][k] = float32(float64(Y2[ch][k]) / (erle * (0.5 + rng.Float64())))
			}
		}
	}
	return &X2, Y2, E2
}

func TestStateFullBandErleEstimator(t *testing.T) {
	for name, cfg := range stTestConfigs() {
		for _, nch := range []int{1, 2, 3} {
			t.Run(fmt.Sprintf("%s/%dch", name, nch), func(t *testing.T) {
				rng := rand.New(rand.NewSource(6))
				g := aec3.NewFullBandErleEstimator(&cfg, nch)
				c := NewStFullBandErleEstimator(&cfg, nch)
				defer c.Free()
				for i := 0; i < 4000; i++ {
					X2, Y2, E2 := stErleInputs(rng, nch, i)
					conv := stRandBools(rng, nch, 0.85)
					if i%1500 == 1499 {
						g.Reset()
						c.Reset()
					}
					g.Update(X2[:], Y2, E2, conv)
					c.Update(X2, Y2, E2, conv)
					stEqF1(t, "fullband erle", g.FullbandErleLog2(), c.FullbandErleLog2())
					stEqQ(t, "qualities", g.GetInstLinearQualityEstimates(), c.Qualities())
				}
			})
		}
	}
}

func TestStateSubbandErleEstimator(t *testing.T) {
	for name, cfg := range stTestConfigs() {
		for _, nch := range []int{1, 2, 3} {
			t.Run(fmt.Sprintf("%s/%dch", name, nch), func(t *testing.T) {
				rng := rand.New(rand.NewSource(7))
				g := aec3.NewSubbandErleEstimator(&cfg, nch)
				c := NewStSubbandErleEstimator(&cfg, nch)
				defer c.Free()
				for i := 0; i < 4000; i++ {
					X2, Y2, E2 := stErleInputs(rng, nch, i)
					conv := stRandBools(rng, nch, 0.85)
					if i%1500 == 1499 {
						g.Reset()
						c.Reset()
					}
					g.Update(X2, Y2, E2, conv)
					c.Update(X2, Y2, E2, conv)
					stEqSpectra(t, "erle", g.Erle(false), c.Get(0))
					stEqSpectra(t, "erle oc", g.Erle(true), c.Get(1))
					stEqSpectra(t, "erle unbounded", g.ErleUnbounded(), c.Get(2))
					stEqSpectra(t, "erle onsets", g.ErleDuringOnsets(), c.Get(3))
				}
			})
		}
	}
}

func TestStateSignalDependentErleEstimator(t *testing.T) {
	cfgs := stTestConfigs()
	for _, name := range []string{"signal_dependent_erle", "long_filter"} {
		cfg := cfgs[name]
		for _, nch := range [][2]int{{1, 1}, {2, 2}} {
			t.Run(fmt.Sprintf("%s/%dx%d", name, nch[0], nch[1]), func(t *testing.T) {
				rng := rand.New(rand.NewSource(8))
				lb := cfg.Filter.Refined.LengthBlocks
				sim := newStRenderSim(rng, 40, nch[0], 3)
				defer sim.r.Free()
				path := newStEchoPath(nch[1], lb)
				path.Change(rng, 2, 0.6, 0.5)
				g := aec3.NewSignalDependentErleEstimator(&cfg, nch[1])
				c := NewStSignalDependentErleEstimator(&cfg, nch[1])
				defer c.Free()
				for i := 0; i < 3000; i++ {
					sim.Advance()
					path.Step(rng)
					X2, Y2, E2 := stErleInputs(rng, nch[1], i)
					avg := stRandSpectra(rng, nch[1], 3)
					avgOc := stRandSpectra(rng, nch[1], 3)
					conv := stRandBools(rng, nch[1], 0.85)
					if i%1000 == 999 {
						g.Reset()
						c.Reset()
					}
					g.Update(sim.r.Buffer, path.H2, X2, Y2, E2, avg, avgOc, conv)
					c.Update(sim.r, path.H2, X2, Y2, E2, avg, avgOc, conv)
					stEqSpectra(t, "erle", g.Erle(false), c.Erle(false))
					stEqSpectra(t, "erle oc", g.Erle(true), c.Erle(true))
				}
			})
		}
	}
}

func TestStateErleEstimator(t *testing.T) {
	for name, cfg := range stTestConfigs() {
		for _, nch := range [][2]int{{1, 1}, {2, 2}, {2, 3}} {
			t.Run(fmt.Sprintf("%s/%dx%d", name, nch[0], nch[1]), func(t *testing.T) {
				rng := rand.New(rand.NewSource(9))
				lb := cfg.Filter.Refined.LengthBlocks
				sim := newStRenderSim(rng, 40, nch[0], 3)
				defer sim.r.Free()
				path := newStEchoPath(nch[1], lb)
				path.Change(rng, 2, 0.6, 0.5)
				g := aec3.NewErleEstimator(50, &cfg, nch[1])
				c := NewStErleEstimator(&cfg, 50, nch[1])
				defer c.Free()
				for i := 0; i < 3000; i++ {
					sim.Advance()
					path.Step(rng)
					X2, Y2, E2 := stErleInputs(rng, nch[1], i)
					conv := stRandBools(rng, nch[1], 0.85)
					if i%900 == 899 {
						d := rng.Intn(2) == 0
						g.Reset(d)
						c.Reset(d)
					}
					g.Update(sim.r.Buffer, path.H2, X2, Y2, E2, conv)
					c.Update(sim.r, path.H2, X2, Y2, E2, conv)
					stEqSpectra(t, "erle", g.Erle(false), c.Get(0))
					stEqSpectra(t, "erle oc", g.Erle(true), c.Get(1))
					stEqSpectra(t, "erle unbounded", g.ErleUnbounded(), c.Get(2))
					stEqSpectra(t, "erle onsets", g.ErleDuringOnsets(), c.Get(3))
					stEqF1(t, "fullband", g.FullbandErleLog2(), c.FullbandErleLog2())
					stEqQ(t, "qualities", g.GetInstLinearQualityEstimates(), c.Qualities())
				}
			})
		}
	}
}

func TestStateStationarityEstimator(t *testing.T) {
	for _, nch := range []int{1, 2} {
		rng := rand.New(rand.NewSource(10))
		sim := newStRenderSim(rng, 40, nch, 4)
		g := aec3.NewStationarityEstimator()
		c := NewStStationarityEstimator()
		for i := 0; i < 3000; i++ {
			sim.Advance()
			spec := sim.r.Buffer.Spectrum(0)
			g.UpdateNoiseEstimator(spec)
			c.UpdateNoiseEstimator(spec)
			var reverb [stK]float32
			for k := range reverb {
				reverb[k] = float32(rng.ExpFloat64() * 1e6)
			}
			idx := rng.Intn(40)
			look := rng.Intn(16)
			g.UpdateStationarityFlags(sim.r.Spectra, reverb[:], idx, look)
			c.UpdateStationarityFlags(sim.r, &reverb, idx, look)
			bands, block := c.Flags()
			for k := range bands {
				stCheck(t, fmt.Sprintf("band %d stationary", k), g.IsBandStationary(k), bands[k])
			}
			stCheck(t, "block stationary", g.IsBlockStationary(), block)
		}
		sim.r.Free()
		c.Free()
	}
}

func TestStateEchoAudibility(t *testing.T) {
	for _, atInit := range []bool{false, true} {
		for _, nch := range []int{1, 2} {
			rng := rand.New(rand.NewSource(11))
			sim := newStRenderSim(rng, 40, nch, 4)
			g := aec3.NewEchoAudibility(atInit)
			c := NewStEchoAudibility(atInit)
			for i := 0; i < 3000; i++ {
				// Sometimes insert several blocks between updates.
				for n := rng.Intn(3); n >= 0; n-- {
					sim.Advance()
				}
				var reverb [stK]float32
				for k := range reverb {
					reverb[k] = float32(rng.ExpFloat64() * 1e6)
				}
				delay := rng.Intn(6)
				ext := i > 500
				g.Update(sim.r.Buffer, reverb[:], delay, ext)
				c.Update(sim.r, &reverb, delay, ext)
				for _, conv := range []bool{false, true} {
					var sg [stK]float32
					g.GetResidualEchoScaling(conv, sg[:])
					sc, blk := c.Get(conv)
					stEqF(t, "scaling", sg[:], sc[:])
					stCheck(t, "block stationary", g.IsBlockStationary(), blk)
				}
			}
			sim.r.Free()
			c.Free()
		}
	}
}

func TestStateFilterAnalyzer(t *testing.T) {
	for name, cfg := range stTestConfigs() {
		for _, nch := range [][2]int{{1, 1}, {2, 2}, {2, 3}} {
			t.Run(fmt.Sprintf("%s/%dx%d", name, nch[0], nch[1]), func(t *testing.T) {
				rng := rand.New(rand.NewSource(12))
				lb := cfg.Filter.Refined.LengthBlocks
				sim := newStRenderSim(rng, 40, nch[0], 3)
				defer sim.r.Free()
				path := newStEchoPath(nch[1], lb)
				path.Change(rng, 2, 0.6, 0.5)
				g := aec3.NewFilterAnalyzer(&cfg, nch[1])
				c := NewStFilterAnalyzer(&cfg, nch[1])
				defer c.Free()
				for i := 0; i < 4000; i++ {
					if i%1300 == 1299 {
						path.Change(rng, rng.Intn(lb-2), 0.3+0.6*rng.Float64(), 0.05+rng.Float64())
						g.Reset()
						c.Reset()
					}
					sim.Advance()
					path.Step(rng)
					filters := path.H
					if i >= 2000 && i < 2400 {
						// Shorter (initial) filters, as during the initial phase.
						filters = make([][]float32, nch[1])
						for ch := range filters {
							filters[ch] = path.H[ch][:cfg.Filter.RefinedInitial.LengthBlocks*aec3.FftLengthBy2]
						}
					}
					cg, gg := g.Update(filters, sim.r.Buffer)
					cc, gc := c.Update(filters, sim.r)
					stCheck(t, "consistent", cg, cc)
					stEqF1(t, "gain", gg, gc)
					delays, minDelay, lengthBlocks, adjusted := c.Get(len(filters[0]))
					stCheck(t, "delays", g.FilterDelaysBlocks(), delays)
					stCheck(t, "min delay", g.MinFilterDelayBlocks(), minDelay)
					stCheck(t, "length", g.FilterLengthBlocks(), lengthBlocks)
					for ch := range adjusted {
						stEqF(t, "adjusted", g.GetAdjustedFilters()[ch], adjusted[ch])
					}
				}
			})
		}
	}
}

func TestStateSubtractorOutputAnalyzer(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for _, nch := range []int{1, 3} {
		g := aec3.NewSubtractorOutputAnalyzer(nch)
		c := NewStSubtractorOutputAnalyzer(nch)
		so := make([]aec3.SubtractorOutput, nch)
		for i := 0; i < 5000; i++ {
			for ch := range so {
				y2 := float32(math.Exp(rng.Float64()*14) * 10)
				so[ch].Y2 = y2
				so[ch].E2RefinedSum = y2 * float32(rng.Float64()*2)
				so[ch].E2CoarseSum = y2 * float32(rng.Float64()*2)
			}
			if i%100 == 0 {
				g.HandleEchoPathChange()
				c.HandleEchoPathChange()
			}
			a, b, d := g.Update(so)
			out, conv := c.Update(so)
			stCheck(t, "flags", [3]bool{a, b, d}, out)
			stCheck(t, "converged", g.ConvergedFilters(), conv)
		}
		c.Free()
	}
}

func TestStateTransparentMode(t *testing.T) {
	for name, cfg := range stTestConfigs() {
		for _, hmm := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/hmm=%v", name, hmm), func(t *testing.T) {
				rng := rand.New(rand.NewSource(14))
				var g aec3.TransparentMode
				if hmm && !cfg.EpStrength.BoundedErl {
					g = aec3.NewTransparentModeHmm()
				} else {
					g = aec3.CreateTransparentMode(&cfg)
				}
				c := NewStTransparentMode(&cfg, hmm)
				if (g == nil) != (c == nil) {
					t.Fatalf("classifier presence: go %v c++ %v", g != nil, c != nil)
				}
				if c == nil {
					return
				}
				defer c.Free()
				// Long stretches of each regime so both classifiers toggle.
				var conv, coarse, consistent, diverged, active, sat bool
				delay := 0
				for i := 0; i < 40000; i++ {
					if i%2000 == 0 {
						regime := rng.Intn(3)
						conv = regime == 0
						coarse = regime != 2
						diverged = regime == 2
					}
					if rng.Intn(20) == 0 {
						active = rng.Intn(4) != 0
						sat = rng.Intn(30) == 0
						consistent = rng.Intn(3) == 0
						delay = rng.Intn(8)
					}
					cv := conv && rng.Intn(10) != 0
					cc := coarse && rng.Intn(10) != 0
					if i%9000 == 8999 {
						g.Reset()
						c.Reset()
					}
					g.Update(delay, consistent, cv, cc, diverged, active, sat)
					stCheck(t, "active", g.Active(), c.Update(delay, consistent, cv, cc, diverged, active, sat))
				}
			})
		}
	}
}

// stAecScenario runs AecState and ResidualEchoEstimator end to end against
// the C++ pair over synthetic render, converging filters, capture = echo +
// noise, echo path changes and saturation.
func stAecScenario(t *testing.T, cfg aec3.EchoCanceller3Config, numRender, numCapture int, hmm bool, blocks int, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	lb := cfg.Filter.Refined.LengthBlocks
	sim := newStRenderSim(rng, 40, numRender, 5)
	defer sim.r.Free()
	path := newStEchoPath(numCapture, lb)
	path.Change(rng, 2, 0.55, 0.4)

	var tm aec3.TransparentMode
	if hmm && !cfg.EpStrength.BoundedErl {
		tm = aec3.NewTransparentModeHmm()
	} else {
		tm = aec3.CreateTransparentMode(&cfg)
	}
	ga := aec3.NewAecStateWithTransparentMode(&cfg, numCapture, tm)
	ca := NewStAecState(&cfg, numCapture, hmm)
	defer ca.Free()
	gr := aec3.NewResidualEchoEstimator(&cfg, numRender)
	cr := NewStResidualEchoEstimator(&cfg, numRender)
	defer cr.Free()

	E2 := make([][stK]float32, numCapture)
	Y2 := make([][stK]float32, numCapture)
	S2 := make([][stK]float32, numCapture)
	so := make([]aec3.SubtractorOutput, numCapture)
	R2g := make([][stK]float32, numCapture)
	R2ug := make([][stK]float32, numCapture)
	R2c := make([][stK]float32, numCapture)
	R2uc := make([][stK]float32, numCapture)

	var ext *aec3.DelayEstimate
	saturatedUntil := -1
	nearendUntil := -1
	for i := 0; i < blocks; i++ {
		sim.Advance()
		if i > 0 && i%1700 == 0 {
			// Echo path change: new delay, decay and gain.
			path.Change(rng, 1+rng.Intn(lb-5), 0.3+0.6*rng.Float64(), 0.05+0.6*rng.Float64())
			v := aec3.EchoPathVariability{GainChange: rng.Intn(2) == 0, DelayChange: aec3.DelayAdjustment(1 + rng.Intn(2))}
			if i%3400 == 0 {
				v.DelayChange = aec3.DelayAdjustmentNone
				v.GainChange = true
			}
			ga.HandleEchoPathChange(v)
			ca.HandleEchoPathChange(v)
			if ext != nil {
				ext = &aec3.DelayEstimate{Quality: aec3.DelayQualityRefined, Delay: ext.Delay + 1 + rng.Intn(3)}
			}
		}
		if i == 150 {
			ext = &aec3.DelayEstimate{Quality: aec3.DelayQualityCoarse, Delay: 5}
		}
		if rng.Intn(600) == 0 {
			saturatedUntil = i + 5 + rng.Intn(40)
		}
		if rng.Intn(500) == 0 {
			nearendUntil = i + 20 + rng.Intn(200)
		}
		saturated := i < saturatedUntil
		dominantNearend := i < nearendUntil
		path.Step(rng)

		X2 := sim.r.Buffer.Spectrum(path.peakBlock)
		conv := 1 - math.Exp(-float64(path.since)/400)
		for ch := 0; ch < numCapture; ch++ {
			var ySum, eSum float64
			for k := 0; k < stK; k++ {
				var x float64
				for rc := 0; rc < numRender; rc++ {
					x += float64(X2[rc][k])
				}
				echo := x * path.gain * path.gain * 0.3
				noise := 2e5 * (1 + rng.Float64())
				if dominantNearend {
					noise += 1e9 * rng.Float64()
				}
				y := echo + noise
				resid := echo*(1-conv*0.97) + noise
				Y2[ch][k] = float32(y)
				E2[ch][k] = float32(resid * (0.8 + 0.4*rng.Float64()))
				S2[ch][k] = float32(echo * conv * (0.9 + 0.2*rng.Float64()))
				ySum += y
				eSum += resid
			}
			// Time-domain energies of one block (64 samples) from the 65-bin
			// spectra of a 128-point transform.
			so[ch].Y2 = float32(ySum / 128)
			so[ch].E2RefinedSum = float32(eSum / 128 * (0.9 + 0.2*rng.Float64()))
			coarse := eSum / 128 * (1.3 - 0.6*conv)
			if path.since < 60 {
				coarse = ySum / 128 * 1.8
				so[ch].E2RefinedSum = float32(coarse)
			}
			so[ch].E2CoarseSum = float32(coarse)
			peak := float32(math.Sqrt(ySum/128/64) * 3)
			if saturated {
				peak = 25000 + float32(rng.Float64()*10000)
			}
			so[ch].SRefinedMaxAbs = peak * float32(0.8+0.3*rng.Float64())
			so[ch].SCoarseMaxAbs = peak * float32(0.8+0.3*rng.Float64())
		}

		ga.UpdateCaptureSaturation(saturated)
		ca.UpdateCaptureSaturation(saturated)
		ga.Update(ext, path.H2, path.H, sim.r.Buffer, E2, Y2, so)
		ca.Update(ext, path.H2, path.H, sim.r, E2, Y2, so)

		want := ca.Get()
		at := func(s string) string { return fmt.Sprintf("block %d: %s", i, s) }
		stEqSpectra(t, at("Erle(false)"), ga.Erle(false), want.Erle)
		stEqSpectra(t, at("Erle(true)"), ga.Erle(true), want.ErleOc)
		stEqSpectra(t, at("ErleUnbounded"), ga.ErleUnbounded(), want.ErleUnbounded)
		stEqF(t, at("Erl"), ga.Erl()[:], want.Erl[:])
		var scaling [stK]float32
		ga.GetResidualEchoScaling(scaling[:])
		stEqF(t, at("GetResidualEchoScaling"), scaling[:], want.ResidualScaling[:])
		stEqF(t, at("GetReverbFrequencyResponse"), ga.GetReverbFrequencyResponse()[:], want.ReverbFreqResp[:])
		stEqF1(t, at("ErlTimeDomain"), ga.ErlTimeDomain(), want.ErlTimeDomain)
		stEqF1(t, at("FullBandErleLog2"), ga.FullBandErleLog2(), want.FullbandErleLog2)
		stEqF1(t, at("ReverbDecay(false)"), ga.ReverbDecay(false), want.ReverbDecay)
		stEqF1(t, at("ReverbDecay(true)"), ga.ReverbDecay(true), want.ReverbDecayMild)
		stCheck(t, at("UsableLinearEstimate"), ga.UsableLinearEstimate(), want.UsableLinearEstimate)
		stCheck(t, at("UseLinearFilterOutput"), ga.UseLinearFilterOutput(), want.UseLinearFilterOutput)
		stCheck(t, at("ActiveRender"), ga.ActiveRender(), want.ActiveRender)
		stCheck(t, at("UseStationarityProperties"), ga.UseStationarityProperties(), want.UseStationarityProperties)
		stCheck(t, at("MinDirectPathFilterDelay"), ga.MinDirectPathFilterDelay(), want.MinDirectPathFilterDelay)
		stCheck(t, at("SaturatedCapture"), ga.SaturatedCapture(), want.SaturatedCapture)
		stCheck(t, at("SaturatedEcho"), ga.SaturatedEcho(), want.SaturatedEcho)
		stCheck(t, at("TransparentModeActive"), ga.TransparentModeActive(), want.TransparentModeActive)
		stCheck(t, at("TransitionTriggered"), ga.TransitionTriggered(), want.TransitionTriggered)
		stCheck(t, at("FilterLengthBlocks"), ga.FilterLengthBlocks(), want.FilterLengthBlocks)

		gr.Estimate(ga, sim.r.Buffer, S2, Y2, dominantNearend, R2g, R2ug)
		cr.Estimate(ca, sim.r, S2, Y2, dominantNearend, R2c, R2uc)
		stEqSpectra(t, at("R2"), R2g, R2c)
		stEqSpectra(t, at("R2 unbounded"), R2ug, R2uc)

		stTrackCoverage(want)
	}
}

var stCoverage = map[string]int{}

func stTrackCoverage(o StAecStateOut) {
	inc := func(k string, b bool) {
		if b {
			stCoverage[k]++
		}
	}
	inc("usable_linear", o.UsableLinearEstimate)
	inc("nonlinear", !o.UsableLinearEstimate)
	inc("active_render", o.ActiveRender)
	inc("saturated_echo", o.SaturatedEcho)
	inc("transparent", o.TransparentModeActive)
	inc("transition", o.TransitionTriggered)
	inc("decay_moved", o.ReverbDecay != 0.83 && o.ReverbDecay != 0.9)
}

func TestStateAecStateEndToEnd(t *testing.T) {
	for name, cfg := range stTestConfigs() {
		for _, ch := range [][2]int{{1, 1}, {2, 2}, {1, 2}, {2, 3}} {
			for _, hmm := range []bool{false, true} {
				if hmm && ch != [2]int{1, 1} {
					continue
				}
				t.Run(fmt.Sprintf("%s/%dx%d/hmm=%v", name, ch[0], ch[1], hmm), func(t *testing.T) {
					stAecScenario(t, cfg, ch[0], ch[1], hmm, 6000, 15)
				})
			}
		}
	}
	t.Logf("coverage (blocks): %v", stCoverage)
}
