//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

const sgK = aec3.FftLengthBy2Plus1

func sgainTestConfigs() map[string]aec3.EchoCanceller3Config {
	out := map[string]aec3.EchoCanceller3Config{}
	def := aec3.DefaultEchoCanceller3Config()
	out["default"] = def

	c := def
	c.Suppressor.UseSubbandNearendDetection = true
	c.Suppressor.SubbandNearendDetection = aec3.SubbandNearendDetectionConfig{
		NearendAverageBlocks: 3,
		Subband1:             aec3.SubbandRegion{Low: 1, High: 4},
		Subband2:             aec3.SubbandRegion{Low: 20, High: 30},
		NearendThreshold:     2,
		SnrThreshold:         10,
	}
	out["subband_nearend"] = c

	c = def
	c.Suppressor.ConservativeHfSuppression = true
	c.Suppressor.HighBandsSuppression.MaxGainDuringEcho = 0.5
	c.Suppressor.HighBandsSuppression.EnrThreshold = 2
	c.Suppressor.DominantNearendDetection.UseUnboundedEchoSpectrum = false
	out["conservative_hf"] = c

	c = def
	c.Suppressor.LfSmoothingDuringInitialPhase = false
	c.Suppressor.LastPermanentLfSmoothingBand = 2
	c.Suppressor.LastLfSmoothingBand = 8
	c.Suppressor.NearendAverageBlocks = 1
	c.Suppressor.DominantNearendDetection.UseDuringInitialPhase = false
	out["lf_smoothing"] = c

	c = def
	c.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 25
	c.Suppressor.HighBandsSuppression.AntiHowlingGain = 0.01
	c.Suppressor.HighBandsSuppression.MaxGainDuringEcho = 0.3
	c.Suppressor.HighBandsSuppression.EnrThreshold = 0.5
	out["high_bands"] = c

	c = def
	c.Suppressor.NormalTuning = aec3.Tuning{
		MaskLf:       aec3.MaskingThresholds{EnrTransparent: .2, EnrSuppress: .5, EmrTransparent: .25},
		MaskHf:       aec3.MaskingThresholds{EnrTransparent: .05, EnrSuppress: .2, EmrTransparent: .4},
		MaxIncFactor: 1.7, MaxDecFactorLf: 0.4,
	}
	c.Suppressor.LastLfBand = 3
	c.Suppressor.FirstHfBand = 17
	c.Suppressor.NearendAverageBlocks = 7
	c.Suppressor.FloorFirstIncrease = 0.001
	c.EchoAudibility.AudibilityThresholdLf = 5
	c.EchoAudibility.AudibilityThresholdMf = 20
	c.EchoAudibility.AudibilityThresholdHf = 3
	c.EchoAudibility.LowRenderLimit = 300
	c.EchoAudibility.NormalRenderLimit = 50
	c.Filter.ConfigChangeDurationBlocks = 100
	out["tuning"] = c
	return out
}

// sgLowNoiseRef replicates LowNoiseRenderDetector for coverage accounting.
type sgLowNoiseRef struct{ avg float32 }

func (d *sgLowNoiseRef) detect(render [][]float32) bool {
	var sum, mx float32
	for _, x := range render {
		for _, v := range x {
			x2 := float32(v * v)
			sum += x2
			mx = max(mx, x2)
		}
	}
	sum /= float32(len(render))
	low := d.avg < 50*50*64 && mx < 3*d.avg
	d.avg = float32(d.avg*0.9) + float32(sum*0.1)
	return low
}

var sgainCoverage = map[string]int{}

type sgainCase struct {
	cfgName    string
	cfg        aec3.EchoCanceller3Config
	opt        aec3.Aec3Optimization
	rate       int
	numRender  int
	numCapture int
	clockDrift int // 0 off, 1 on, 2 toggling
	initial    bool
	blocks     int
	seed       int64
}

func sgainScenario(t *testing.T, tc sgainCase) {
	rng := rand.New(rand.NewSource(tc.seed))
	cfg := tc.cfg
	nr, nc := tc.numRender, tc.numCapture
	numBands := aec3.NumBandsForRate(tc.rate)
	const size, renderDelay = 40, 5
	h := NewSgainHarness(&cfg, tc.opt, tc.rate, nr, nc, size)
	defer h.Free()
	initState := tc.initial
	if !tc.initial {
		h.SetInitialState(false)
	}

	lb := cfg.Filter.Refined.LengthBlocks
	path := newStEchoPath(nc, lb)
	path.Change(rng, 2, 0.55, 0.4)

	E2 := make([][sgK]float32, nc)
	Y2 := make([][sgK]float32, nc)
	S2 := make([][sgK]float32, nc)
	R2 := make([][sgK]float32, nc)
	R2u := make([][sgK]float32, nc)
	N2 := make([][sgK]float32, nc)
	so := make([]aec3.SubtractorOutput, nc)
	render := make([]float32, numBands*nr*aec3.BlockSize)
	renderGo := aec3.NewBlock(numBands, nr)
	bandViews := make([][]float32, nr)
	var X2 [sgK]float32
	var lowGo, lowC [sgK]float32
	write := 0
	lowRef := sgLowNoiseRef{avg: 32768 * 32768}

	var ext *aec3.DelayEstimate
	saturatedUntil, nearendUntil, quietUntil, howlUntil, peakUntil, silentUntil := -1, -1, -1, -1, -1, -1
	clockDrift := tc.clockDrift == 1
	for i := 0; i < tc.blocks; i++ {
		// Scenario phases.
		if rng.Intn(600) == 0 {
			saturatedUntil = i + 5 + rng.Intn(40)
		}
		if rng.Intn(350) == 0 {
			nearendUntil = i + 30 + rng.Intn(200)
		}
		if rng.Intn(500) == 0 {
			quietUntil = i + 40 + rng.Intn(150)
		}
		if numBands > 1 && rng.Intn(400) == 0 {
			howlUntil = i + 20 + rng.Intn(80)
		}
		if rng.Intn(700) == 0 {
			peakUntil = i + 10 + rng.Intn(40)
		}
		if rng.Intn(900) == 0 {
			silentUntil = i + 20 + rng.Intn(60)
		}
		if tc.clockDrift == 2 && rng.Intn(300) == 0 {
			clockDrift = !clockDrift
		}
		if tc.initial && i == 400+int(tc.seed%7)*50 {
			initState = false
			h.SetInitialState(false)
		}
		if tc.initial && i > 1000 && rng.Intn(1500) == 0 {
			initState = rng.Intn(2) == 0
			h.SetInitialState(initState)
		}
		saturated := i < saturatedUntil
		nearend := i < nearendUntil
		quiet := i < quietUntil
		howl := i < howlUntil
		peak := i < peakUntil
		silent := i < silentUntil

		// Render signal.
		tm := float64(i)
		write = (write - 1 + size) % size
		for ch := 0; ch < nr; ch++ {
			level := 3000 * (0.5 + 0.5*math.Sin(2*math.Pi*tm/(310+37*float64(ch)))) *
				(0.6 + 0.4*math.Sin(2*math.Pi*tm/2700))
			switch {
			case silent:
				level = 0
			case quiet:
				level = 8 + 4*math.Sin(tm/30)
			case howl:
				level = 150
			case peak:
				level = 2000
			}
			for b := 0; b < numBands; b++ {
				bl := level
				if b > 0 {
					bl = level * 0.3
					if howl {
						bl = 6000
					}
				}
				v := render[(b*nr+ch)*aec3.BlockSize : (b*nr+ch+1)*aec3.BlockSize]
				for n := range v {
					if peak && b == 0 {
						v[n] = float32(bl * math.Sin(2*math.Pi*58*float64(n+64*i)/128))
					} else {
						v[n] = float32(bl * rng.NormFloat64())
					}
				}
				copy(renderGo.View(b, ch), v)
			}
			h.SetRenderBlock(write, ch, render[ch*aec3.BlockSize:(ch+1)*aec3.BlockSize])
			for k := range X2 {
				g, q := rng.NormFloat64(), rng.NormFloat64()
				X2[k] = float32(128 * level * level / (1 + float64(k)/12) * 0.5 * (g*g + q*q))
				if peak {
					X2[k] *= 1e-4
				}
			}
			if peak {
				X2[58] = float32(128 * level * level * 64)
			}
			h.SetRenderSpectrum(write, ch, &X2)
			bandViews[ch] = renderGo.View(0, ch)
		}
		h.SetIndices(write, (write+renderDelay)%size)

		// Echo path and AecState.
		if i > 0 && i%1700 == 0 {
			path.Change(rng, 1+rng.Intn(lb-5), 0.3+0.6*rng.Float64(), 0.05+0.6*rng.Float64())
			v := aec3.EchoPathVariability{GainChange: rng.Intn(2) == 0, DelayChange: aec3.DelayAdjustment(1 + rng.Intn(2))}
			h.HandleEchoPathChange(v)
			if ext != nil {
				ext = &aec3.DelayEstimate{Quality: aec3.DelayQualityRefined, Delay: ext.Delay + 1 + rng.Intn(3)}
			}
		}
		if i == 150 {
			ext = &aec3.DelayEstimate{Quality: aec3.DelayQualityCoarse, Delay: 5}
		}
		path.Step(rng)
		Xd := h.Buffer.Spectrum(path.peakBlock)
		conv := 1 - math.Exp(-float64(path.since)/400)
		for ch := 0; ch < nc; ch++ {
			var ySum, eSum float64
			for k := 0; k < sgK; k++ {
				var x float64
				for rc := 0; rc < nr; rc++ {
					x += float64(Xd[rc][k])
				}
				echo := x * path.gain * path.gain * 0.3
				noise := 2e4 * (1 + rng.Float64())
				if nearend {
					noise += 1e9 * rng.Float64() * (1 - 0.6*float64(k)/sgK)
				}
				if silent {
					echo, noise = 0, 0
				}
				y := echo + noise
				resid := echo*(1-conv*0.97) + noise*0.01
				Y2[ch][k] = float32(y)
				E2[ch][k] = float32(echo*(1-conv*0.97) + noise)
				S2[ch][k] = float32(echo * conv * (0.9 + 0.2*rng.Float64()))
				R2[ch][k] = float32(resid * (0.8 + 0.4*rng.Float64()))
				R2u[ch][k] = float32(float64(R2[ch][k]) * (1 + 3*rng.Float64()))
				N2[ch][k] = float32(1e4 * (0.5 + rng.Float64()))
				if silent {
					N2[ch][k] = 0
				}
				ySum += y
				eSum += resid
			}
			so[ch].Y2 = float32(ySum / 128)
			so[ch].E2RefinedSum = float32(eSum / 128 * (0.9 + 0.2*rng.Float64()))
			coarse := eSum / 128 * (1.3 - 0.6*conv)
			if path.since < 60 {
				coarse = ySum / 128 * 1.8
				so[ch].E2RefinedSum = float32(coarse)
			}
			so[ch].E2CoarseSum = float32(coarse)
			pk := float32(math.Sqrt(ySum/128/64) * 3)
			if saturated {
				pk = 25000 + float32(rng.Float64()*10000)
			}
			so[ch].SRefinedMaxAbs = pk * float32(0.8+0.3*rng.Float64())
			so[ch].SCoarseMaxAbs = pk * float32(0.8+0.3*rng.Float64())
		}
		h.UpdateCaptureSaturation(saturated)
		h.AecUpdate(ext, path.H2, path.H, E2, Y2, so)

		delay := h.Aec.MinDirectPathFilterDelay()
		if cd := h.CMinDirectPathFilterDelay(); cd != delay {
			t.Fatalf("block %d: MinDirectPathFilterDelay go %d c %d", i, delay, cd)
		}
		hasDelay := rng.Intn(20) != 0
		h.RsaUpdate(delay, hasDelay)
		gb, gok := h.Rsa.NarrowPeakBand()
		cb, cok := h.CNarrowPeakBand()
		if gok != cok || (gok && gb != cb) {
			t.Fatalf("block %d: NarrowPeakBand go (%d,%v) c (%d,%v)", i, gb, gok, cb, cok)
		}
		if h.Aec.SaturatedEcho() != h.CSaturatedEcho() {
			t.Fatalf("block %d: SaturatedEcho mismatch", i)
		}

		// Suppression gain.
		var highGo float32
		h.Gain.GetGain(Y2, S2, R2, R2u, N2, h.Rsa, h.Aec, &renderGo, clockDrift, &highGo, &lowGo)
		highC, dnC := h.CGetGain(Y2, S2, R2, R2u, N2, render, numBands, clockDrift, &lowC)
		if dnGo := h.Gain.IsDominantNearend(); dnGo != dnC {
			t.Fatalf("block %d: IsDominantNearend go %v c %v", i, dnGo, dnC)
		}
		for k := range lowGo {
			if math.Float32bits(lowGo[k]) != math.Float32bits(lowC[k]) {
				t.Fatalf("block %d: low_band_gain[%d] go %g (%08x) c %g (%08x)", i, k, lowGo[k],
					math.Float32bits(lowGo[k]), lowC[k], math.Float32bits(lowC[k]))
			}
		}
		if math.Float32bits(highGo) != math.Float32bits(highC) {
			t.Fatalf("block %d: high_bands_gain go %g c %g", i, highGo, highC)
		}

		// Coverage accounting.
		inc := func(k string, b bool) {
			if b {
				sgainCoverage[k]++
			}
		}
		lowNoise := lowRef.detect(bandViews)
		sat := h.Aec.SaturatedEcho()
		inc("dominant_nearend", dnC)
		inc("dominant_nearend/"+tc.cfgName, dnC)
		inc("saturated_echo", sat)
		inc("low_noise_render", lowNoise)
		inc("clock_drift", clockDrift)
		inc("drift_forces_hf_limit", dnC && clockDrift && !cfg.Suppressor.ConservativeHfSuppression)
		inc("initial_no_lf_smoothing", initState && !cfg.Suppressor.LfSmoothingDuringInitialPhase)
		inc("hf_limit_skipped", dnC && !clockDrift && !cfg.Suppressor.ConservativeHfSuppression)
		inc("single_band", numBands == 1)
		minLow := float32(1)
		for _, g := range lowC {
			minLow = min(minLow, g)
		}
		inc("gain_below_one", minLow < 1)
		if numBands > 1 {
			inc("narrow_peak_return", gok && gb > sgK-10 && highC == 0.001)
			inc("saturated_high", sat && !(gok && gb > sgK-10))
			if !sat && !(gok && gb > sgK-10) {
				var lowE, highE float32
				for ch := 0; ch < nr; ch++ {
					var e float32
					for _, v := range renderGo.View(0, ch) {
						e += float32(v * v)
					}
					lowE = max(lowE, e)
					for b := 1; b < numBands; b++ {
						var e float32
						for _, v := range renderGo.View(b, ch) {
							e += float32(v * v)
						}
						highE = max(highE, e)
					}
				}
				thr := aec3.BlockSize * cfg.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold
				inc("anti_howling", highE >= max(lowE, thr))
				mg := cfg.Suppressor.HighBandsSuppression.MaxGainDuringEcho
				inc("gain_bound", mg < 1 && !dnC && highC == mg)
			}
		}
	}
}

func TestSgainSuppressionGain(t *testing.T) {
	cfgs := sgainTestConfigs()
	names := make([]string, 0, len(cfgs))
	for n := range cfgs {
		names = append(names, n)
	}
	sort.Strings(names)
	seed := int64(1)
	for _, name := range names {
		for ri, rate := range []int{16000, 32000, 48000} {
			for _, nc := range []int{1, 2} {
				for _, opt := range []aec3.Aec3Optimization{aec3.OptNone, aec3.OptNeon} {
					seed++
					nr := nc
					if ri == 1 {
						nr = 3 - nc
					}
					tc := sgainCase{cfgName: name, cfg: cfgs[name], opt: opt, rate: rate, numRender: nr,
						numCapture: nc, clockDrift: int(seed % 3), initial: seed%4 != 0, blocks: 3000, seed: seed}
					t.Run(fmt.Sprintf("%s/%d/%dx%d/opt%d/drift%d/init%v", name, rate, nr, nc, opt, tc.clockDrift,
						tc.initial), func(t *testing.T) {
						sgainScenario(t, tc)
					})
				}
			}
		}
	}
	t.Logf("coverage (blocks): %v", sgainCoverage)
	for _, k := range []string{
		"dominant_nearend", "saturated_echo", "low_noise_render", "clock_drift", "hf_limit_skipped",
		"single_band", "gain_below_one", "drift_forces_hf_limit", "initial_no_lf_smoothing", "narrow_peak_return", "saturated_high",
		"anti_howling", "gain_bound",
	} {
		if sgainCoverage[k] == 0 {
			t.Errorf("path %q never exercised", k)
		}
	}
	for _, n := range names {
		if sgainCoverage["dominant_nearend/"+n] == 0 {
			t.Errorf("dominant nearend never reached with config %q", n)
		}
	}
}
