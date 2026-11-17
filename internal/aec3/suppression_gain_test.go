package aec3

import (
	"math/rand"
	"testing"
)

func sgTestSpectra(rng *rand.Rand, n int, scale float32) [][FftLengthBy2Plus1]float32 {
	out := make([][FftLengthBy2Plus1]float32, n)
	for ch := range out {
		for k := range out[ch] {
			out[ch][k] = scale * rng.Float32()
		}
	}
	return out
}

func TestSuppressionGainGetGainNoAllocs(t *testing.T) {
	for _, subband := range []bool{false, true} {
		cfg := DefaultEchoCanceller3Config()
		cfg.Suppressor.UseSubbandNearendDetection = subband
		const nch = 2
		rng := rand.New(rand.NewSource(1))
		g := NewSuppressionGain(&cfg, DetectOptimization(), 48000, nch)
		aec := NewAecState(&cfg, nch)
		rsa := NewRenderSignalAnalyzer(&cfg)
		render := NewBlock(3, 2)
		for b := 0; b < 3; b++ {
			for ch := 0; ch < 2; ch++ {
				for i, v := range render.View(b, ch) {
					render.View(b, ch)[i] = v + 1000*float32(rng.NormFloat64())
				}
			}
		}
		ne := sgTestSpectra(rng, nch, 1e6)
		echo := sgTestSpectra(rng, nch, 1e6)
		r := sgTestSpectra(rng, nch, 1e6)
		ru := sgTestSpectra(rng, nch, 2e6)
		cn := sgTestSpectra(rng, nch, 1e4)
		var high float32
		var low [FftLengthBy2Plus1]float32
		run := func() {
			g.GetGain(ne, echo, r, ru, cn, rsa, aec, &render, false, &high, &low)
		}
		for i := 0; i < 10; i++ {
			run()
		}
		if n := testing.AllocsPerRun(200, run); n != 0 {
			t.Fatalf("subband=%v: GetGain allocates %v times per call", subband, n)
		}
		for k, v := range low {
			if !(v >= 0 && v <= 1) {
				t.Fatalf("low_band_gain[%d] = %v out of [0,1]", k, v)
			}
		}
		if !(high >= 0 && high <= 1) {
			t.Fatalf("high_bands_gain = %v out of [0,1]", high)
		}
	}
}

func TestSuppressionGainSingleBand(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	g := NewSuppressionGain(&cfg, OptNone, 16000, 1)
	aec := NewAecState(&cfg, 1)
	rsa := NewRenderSignalAnalyzer(&cfg)
	render := NewBlock(1, 1)
	z := make([][FftLengthBy2Plus1]float32, 1)
	var high float32
	var low [FftLengthBy2Plus1]float32
	g.GetGain(z, z, z, z, z, rsa, aec, &render, false, &high, &low)
	if high != 1 {
		t.Fatalf("single band high gain %v, want 1", high)
	}
	for k, v := range low {
		if v != 1 {
			t.Fatalf("silent input low gain[%d] = %v, want 1", k, v)
		}
	}
}
