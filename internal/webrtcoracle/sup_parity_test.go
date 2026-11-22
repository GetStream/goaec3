//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

type supSpectra = [][aec3.FftLengthBy2Plus1]float32

func supCheck(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: got %v (%#x) want %v (%#x)", what, i, got[i], math.Float32bits(got[i]),
				want[i], math.Float32bits(want[i]))
		}
	}
}

func supCheckFft(t *testing.T, what string, got, want []aec3.FftData) {
	t.Helper()
	for ch := range got {
		supCheck(t, fmt.Sprintf("%s ch%d re", what, ch), got[ch].Re[:], want[ch].Re[:])
		supCheck(t, fmt.Sprintf("%s ch%d im", what, ch), got[ch].Im[:], want[ch].Im[:])
	}
}

// supOpts are the optimization lanes compared against the oracle: the scalar
// lane, and in the default arm64 build also the NEON lane.
func supOpts() []aec3.Aec3Optimization {
	return []aec3.Aec3Optimization{aec3.OptNone, aec3.OptNeon}
}

// supRandomSpectrum fills a power spectrum with a level that wanders over
// many orders of magnitude, including levels below the noise floor.
func supRandomSpectrum(rng *rand.Rand, s *[aec3.FftLengthBy2Plus1]float32, level float64) {
	for k := range s {
		v := level * math.Exp(rng.NormFloat64()) * (1 + 0.5*math.Sin(float64(k)/5))
		s[k] = float32(v)
	}
}

func TestSupComfortNoiseGeneratorParity(t *testing.T) {
	for _, opt := range supOpts() {
		for _, numCh := range []int{1, 2} {
			for _, floor := range []float32{-96.03406, -60, -130.5, 0} {
				t.Run(fmt.Sprintf("opt%d_ch%d_floor%v", opt, numCh, floor), func(t *testing.T) {
					cfg := aec3.DefaultEchoCanceller3Config()
					cfg.ComfortNoise.NoiseFloorDbfs = floor
					g := aec3.NewComfortNoiseGenerator(&cfg, opt, numCh)
					o := NewSupComfortNoiseGenerator(floor, opt, numCh)
					defer o.Free()
					rng := rand.New(rand.NewSource(int64(numCh)*7 + int64(opt)))
					Y2 := make(supSpectra, numCh)
					lowG, highG := make([]aec3.FftData, numCh), make([]aec3.FftData, numCh)
					lowC, highC := make([]aec3.FftData, numCh), make([]aec3.FftData, numCh)
					// Preset distinct garbage so untouched bins are compared too.
					for ch := range lowG {
						for k := range lowG[ch].Im {
							v := float32(rng.NormFloat64())
							lowG[ch].Im[k], lowC[ch].Im[k] = v, v
							highG[ch].Im[k], highC[ch].Im[k] = -v, -v
						}
					}
					N2C := make(supSpectra, numCh)
					level := 1e4
					for blk := 0; blk < 2500; blk++ {
						if blk%40 == 0 {
							level = math.Pow(10, rng.Float64()*12-2)
						}
						for ch := range Y2 {
							supRandomSpectrum(rng, &Y2[ch], level)
						}
						sat := rng.Intn(20) == 0
						g.Compute(sat, Y2, lowG, highG)
						o.Compute(sat, Y2, lowC, highC, N2C)
						supCheckFft(t, fmt.Sprintf("blk %d low", blk), lowG, lowC)
						supCheckFft(t, fmt.Sprintf("blk %d high", blk), highG, highC)
						n2 := g.NoiseSpectrum()
						for ch := range n2 {
							supCheck(t, fmt.Sprintf("blk %d N2 ch%d", blk, ch), n2[ch][:], N2C[ch][:])
						}
					}
				})
			}
		}
	}
}

// TestSupSuppressionFilterParity compares against the lane the build selects:
// the libwebrtc.a SuppressionFilter (NEON Ooura FFT) in the default build and
// the from-source scalar build (sup_scalar.cc) under purego. The default lane
// depends on the NEON lane of internal/ooura (TestOouraFftBitExact).
func TestSupSuppressionFilterParity(t *testing.T) {
	for _, opt := range supOpts() {
		for _, rate := range []int{16000, 32000, 48000} {
			for _, numCh := range []int{1, 2} {
				t.Run(fmt.Sprintf("opt%d_%d_ch%d", opt, rate, numCh), func(t *testing.T) {
					numBands := aec3.NumBandsForRate(rate)
					f := aec3.NewSuppressionFilter(opt, rate, numCh)
					o := NewSupSuppressionFilter(opt, rate, numCh, purego)
					defer o.Free()
					rng := rand.New(rand.NewSource(int64(rate + numCh + int(opt))))
					cn, cnHigh, E := make([]aec3.FftData, numCh), make([]aec3.FftData, numCh), make([]aec3.FftData, numCh)
					var gain [aec3.FftLengthBy2Plus1]float32
					e := aec3.NewBlock(numBands, numCh)
					eC := make([]float32, numBands*numCh*aec3.BlockSize)
					for blk := 0; blk < 1000; blk++ {
						scale := math.Pow(10, rng.Float64()*6)
						for ch := 0; ch < numCh; ch++ {
							for k := 0; k < aec3.FftLengthBy2Plus1; k++ {
								E[ch].Re[k] = float32(rng.NormFloat64() * scale)
								E[ch].Im[k] = float32(rng.NormFloat64() * scale)
								cn[ch].Re[k] = float32(rng.NormFloat64() * 100)
								cn[ch].Im[k] = float32(rng.NormFloat64() * 100)
								cnHigh[ch].Re[k] = float32(rng.NormFloat64() * 100)
								cnHigh[ch].Im[k] = float32(rng.NormFloat64() * 100)
							}
						}
						for k := range gain {
							gain[k] = float32(rng.Float64())
						}
						if blk%7 == 0 {
							gain[rng.Intn(len(gain))] = 1
						}
						hg := float32(rng.Float64())
						for b := 0; b < numBands; b++ {
							for ch := 0; ch < numCh; ch++ {
								v := e.View(b, ch)
								for i := range v {
									v[i] = float32(rng.NormFloat64() * scale * 0.05)
									eC[(b*numCh+ch)*aec3.BlockSize+i] = v[i]
								}
							}
						}
						f.ApplyGain(cn, cnHigh, &gain, hg, E, &e)
						o.ApplyGain(cn, cnHigh, &gain, hg, E, eC, numBands, numCh)
						for b := 0; b < numBands; b++ {
							for ch := 0; ch < numCh; ch++ {
								i := (b*numCh + ch) * aec3.BlockSize
								supCheck(t, fmt.Sprintf("blk %d band %d ch %d", blk, b, ch), e.View(b, ch),
									eC[i:i+aec3.BlockSize])
							}
						}
					}
				})
			}
		}
	}
}

func supNearendInputs(rng *rand.Rand, ne, echo, noise supSpectra) {
	// Regimes: nearend dominant, echo dominant, noise dominant.
	r := rng.Intn(3)
	for ch := range ne {
		lvl := math.Pow(10, rng.Float64()*6)
		ne0, e0, n0 := lvl, lvl, lvl
		switch r {
		case 0:
			e0, n0 = lvl*0.01*rng.Float64(), lvl*0.001*rng.Float64()
		case 1:
			ne0 = lvl * 0.05 * rng.Float64()
		case 2:
			n0 = lvl * (0.5 + rng.Float64())
		}
		supRandomSpectrum(rng, &ne[ch], ne0)
		supRandomSpectrum(rng, &echo[ch], e0)
		supRandomSpectrum(rng, &noise[ch], n0)
	}
}

func TestSupDominantNearendDetectorParity(t *testing.T) {
	def := aec3.DefaultEchoCanceller3Config().Suppressor.DominantNearendDetection
	alt := def
	alt.EnrThreshold, alt.EnrExitThreshold, alt.SnrThreshold = 0.5, 4, 8
	alt.HoldDuration, alt.TriggerThreshold, alt.UseDuringInitialPhase = 20, 3, false
	for ci, cfg := range []aec3.DominantNearendDetectionConfig{def, alt} {
		for _, numCh := range []int{1, 2, 3} {
			t.Run(fmt.Sprintf("cfg%d_ch%d", ci, numCh), func(t *testing.T) {
				d := aec3.NewDominantNearendDetector(&cfg, numCh)
				o := NewSupDominantNearendDetector(&cfg, numCh)
				defer o.Free()
				rng := rand.New(rand.NewSource(int64(ci*10 + numCh)))
				ne, echo, noise := make(supSpectra, numCh), make(supSpectra, numCh), make(supSpectra, numCh)
				changes := 0
				prev := false
				for blk := 0; blk < 5000; blk++ {
					if blk%(5+rng.Intn(60)) == 0 || blk == 0 {
						supNearendInputs(rng, ne, echo, noise)
					} else {
						for ch := range ne {
							ne[ch][3] *= float32(0.9 + 0.2*rng.Float64())
						}
					}
					initial := blk < 1000 || rng.Intn(10) == 0
					d.Update(ne, echo, noise, initial)
					want := o.Update(ne, echo, noise, initial)
					if got := d.IsNearendState(); got != want {
						t.Fatalf("blk %d: got %v want %v", blk, got, want)
					}
					if want != prev {
						changes++
					}
					prev = want
				}
				if changes < 4 {
					t.Fatalf("only %d state changes; inputs do not exercise the detector", changes)
				}
			})
		}
	}
}

func TestSupSubbandNearendDetectorParity(t *testing.T) {
	def := aec3.DefaultEchoCanceller3Config().Suppressor.SubbandNearendDetection
	alt := aec3.SubbandNearendDetectionConfig{
		NearendAverageBlocks: 4,
		Subband1:             aec3.SubbandRegion{Low: 1, High: 10},
		Subband2:             aec3.SubbandRegion{Low: 20, High: 40},
		NearendThreshold:     1.5,
		SnrThreshold:         2,
	}
	alt2 := alt
	alt2.NearendAverageBlocks = 1
	alt2.Subband1 = aec3.SubbandRegion{Low: 0, High: 64}
	for ci, cfg := range []aec3.SubbandNearendDetectionConfig{def, alt, alt2} {
		for _, numCh := range []int{1, 2} {
			t.Run(fmt.Sprintf("cfg%d_ch%d", ci, numCh), func(t *testing.T) {
				d := aec3.NewSubbandNearendDetector(&cfg, numCh)
				o := NewSupSubbandNearendDetector(&cfg, numCh)
				defer o.Free()
				rng := rand.New(rand.NewSource(int64(ci*10 + numCh + 100)))
				ne, echo, noise := make(supSpectra, numCh), make(supSpectra, numCh), make(supSpectra, numCh)
				trues := 0
				for blk := 0; blk < 5000; blk++ {
					supNearendInputs(rng, ne, echo, noise)
					for ch := range ne {
						tilt := math.Pow(10, rng.Float64()*2-1)
						for k := range ne[ch] {
							ne[ch][k] *= float32(math.Pow(tilt, float64(k)/64))
						}
					}
					d.Update(ne, echo, noise, false)
					want := o.Update(ne, echo, noise, false)
					if got := d.IsNearendState(); got != want {
						t.Fatalf("blk %d: got %v want %v", blk, got, want)
					}
					if want {
						trues++
					}
				}
				if ci > 0 && (trues == 0 || trues == 5000) {
					t.Fatalf("nearend state constant (%d true); inputs do not exercise the detector", trues)
				}
			})
		}
	}
}

func TestSupMultiChannelContentDetectorParity(t *testing.T) {
	type params struct {
		detect     bool
		numCh      int
		threshold  float32
		timeout    int
		hysteresis float32
	}
	cases := []params{
		{true, 2, 0, 300, 2},
		{true, 2, 0.01, 1, 0.13},
		{true, 2, 0, 0, 0.5},
		{true, 2, 0.5, -1, 0},
		{false, 2, 0, 300, 2},
		{false, 1, 0, 300, 2},
		{true, 1, 0, 300, 2},
	}
	for ci, c := range cases {
		t.Run(fmt.Sprintf("case%d", ci), func(t *testing.T) {
			d := aec3.NewMultiChannelContentDetector(c.detect, c.numCh, c.threshold, c.timeout, c.hysteresis)
			o := NewSupMultiChannelContentDetector(c.detect, c.numCh, c.threshold, c.timeout, c.hysteresis)
			defer o.Free()
			rng := rand.New(rand.NewSource(int64(ci)))
			const numBands, n = 2, 160
			frame := make([][][]float32, numBands)
			flat := make([]float32, numBands*c.numCh*n)
			for b := range frame {
				frame[b] = make([][]float32, c.numCh)
				for ch := range frame[b] {
					i := (b*c.numCh + ch) * n
					frame[b][ch] = flat[i : i+n]
				}
			}
			goFrame := aec3.NewFrame(numBands, c.numCh) // the layout of flat
			stereo := false
			changes := 0
			for blk := 0; blk < 12000; blk++ {
				if rng.Intn(150) == 0 {
					stereo = !stereo
				}
				for b := range frame {
					for k := 0; k < n; k++ {
						v := float32(rng.NormFloat64() * 1000)
						for ch := range frame[b] {
							frame[b][ch][k] = v
						}
						if stereo && c.numCh > 1 && rng.Intn(50) == 0 {
							frame[b][1][k] = v + float32(rng.NormFloat64())
						}
					}
				}
				copy(goFrame.Samples(), flat)
				changed := d.UpdateDetection(goFrame)
				wc, wp, wt := o.UpdateDetection(flat, numBands, c.numCh, n)
				if changed != wc || d.IsProperMultiChannelContentDetected() != wp ||
					d.IsTemporaryMultiChannelContentDetected() != wt {
					t.Fatalf("blk %d: got (%v %v %v) want (%v %v %v)", blk, changed,
						d.IsProperMultiChannelContentDetected(), d.IsTemporaryMultiChannelContentDetected(), wc, wp, wt)
				}
				if wc {
					changes++
				}
			}
			t.Logf("%d persistent-state changes", changes)
		})
	}
}

func TestSupConfigSelectorParity(t *testing.T) {
	const marker = 77
	for _, detect := range []bool{false, true} {
		for _, hasMc := range []bool{false, true} {
			for _, numCh := range []int{1, 2} {
				mono := aec3.DefaultEchoCanceller3Config()
				mono.MultiChannel.DetectStereoContent = detect
				var mc *aec3.EchoCanceller3Config
				if hasMc {
					m := mono
					m.Filter.Refined.LengthBlocks = marker
					mc = &m
				}
				s := aec3.NewConfigSelector(&mono, mc, numCh)
				o := NewSupConfigSelector(detect, hasMc, marker, numCh)
				check := func(step string) {
					if got, want := s.ActiveConfig().Filter.Refined.LengthBlocks, o.ActiveMarker(); got != want {
						t.Fatalf("detect=%v mc=%v ch=%d %s: got %d want %d", detect, hasMc, numCh, step, got, want)
					}
				}
				check("ctor")
				for _, u := range []bool{true, false, true, true, false} {
					s.Update(u)
					o.Update(u)
					check(fmt.Sprintf("update(%v)", u))
				}
				o.Free()
			}
		}
	}
}
