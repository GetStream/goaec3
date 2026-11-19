package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// refCascade applies the sections one after the other, each over the whole
// signal (the C++ loop over biquads_).
func refCascade(x, y []float32, biquads []BiQuad) {
	if len(biquads) == 0 {
		copy(y[:len(x)], x)
		return
	}
	dlyApplyBiQuad(x, y, &biquads[0])
	for k := 1; k < len(biquads); k++ {
		dlyApplyBiQuad(y, y, &biquads[k])
	}
}

func refDecimate(d *Decimator, in, out []float32) {
	var x [BlockSize]float32
	refCascade(in[:BlockSize], x[:], d.antiAliasing())
	refCascade(x[:], x[:], d.noiseReduction())
	for j, k := 0, 0; j < BlockSize/d.downSamplingFactor; j, k = j+1, k+d.downSamplingFactor {
		out[j] = x[k]
	}
}

func biquadTestCoefficients(rng *rand.Rand, n int) []BiQuadCoefficients {
	pool := append(append(append(append([]BiQuadCoefficients{}, dlyLowPassFilterDs4[:]...), dlyBandPassFilterDs8[0]),
		dlyHighPassFilter[0]), ec3HighPassFilterCoefficients16kHz[0], ec3HighPassFilterCoefficients48kHz[0])
	c := make([]BiQuadCoefficients, n)
	for i := range c {
		c[i] = pool[rng.Intn(len(pool))]
		if rng.Intn(3) == 0 {
			// A random stable section: poles inside the unit circle.
			r := float32(0.2 + 0.75*rng.Float64())
			th := rng.Float64() * math.Pi
			c[i].A = [2]float32{float32(-2 * float64(r) * math.Cos(th)), r * r}
			c[i].B = [3]float32{float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())}
		}
	}
	return c
}

func biquadTestSignal(rng *rand.Rand, x []float32) {
	scale := []float64{1, 4000, 32768, 1e-3}[rng.Intn(4)]
	for i := range x {
		x[i] = float32(rng.NormFloat64() * scale)
	}
	if len(x) == 0 {
		return
	}
	if rng.Intn(8) == 0 {
		x[rng.Intn(len(x))] = float32(math.Inf(1 - 2*rng.Intn(2)))
	}
	if rng.Intn(8) == 0 {
		x[rng.Intn(len(x))] = float32(math.NaN())
	}
}

// equalBits returns the first index where a and b differ, -1 if none: bit
// for bit, except that any two NaNs match (as in kernCheckExact; the NaN sign
// and payload depend on the operand order the compiler picks for a
// commutative operation on amd64, as they do for the C++).
func equalBits(a, b []float32) int {
	for i := range a {
		if a[i] != a[i] && b[i] != b[i] {
			continue
		}
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	return -1
}

func TestCascadedBiQuadFilterMatchesSectionBySection(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for numSections := 0; numSections <= 9; numSections++ {
		for trial := 0; trial < 20; trial++ {
			coefficients := biquadTestCoefficients(rng, numSections)
			got := NewCascadedBiQuadFilter(coefficients)
			want := NewCascadedBiQuadFilter(coefficients)
			inPlace := rng.Intn(2) == 0
			for block := 0; block < 30; block++ {
				n := []int{0, 1, 3, 16, 64, 160, 161}[rng.Intn(7)]
				x := make([]float32, n)
				biquadTestSignal(rng, x)
				var yGot, yWant []float32
				if inPlace {
					yGot = append([]float32(nil), x...)
					yWant = append([]float32(nil), x...)
					got.ProcessInPlace(yGot)
					refCascade(yWant, yWant, want.biquads)
				} else {
					yGot = make([]float32, n)
					yWant = make([]float32, n)
					got.Process(x, yGot)
					refCascade(x, yWant, want.biquads)
				}
				if i := equalBits(yGot, yWant); i >= 0 {
					t.Fatalf("%d sections, trial %d, block %d, in place %v: y[%d] = %v, want %v",
						numSections, trial, block, inPlace, i, yGot[i], yWant[i])
				}
				for k := range got.biquads {
					g, w := &got.biquads[k], &want.biquads[k]
					if equalBits(append(g.X[:], g.Y[:]...), append(w.X[:], w.Y[:]...)) >= 0 {
						t.Fatalf("%d sections: state of section %d differs", numSections, k)
					}
				}
			}
		}
	}
}

func TestDecimatorMatchesSectionBySection(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for _, factor := range []int{4, 8} {
		got := NewDecimator(factor)
		want := NewDecimator(factor)
		for block := 0; block < 3000; block++ {
			var in [BlockSize]float32
			biquadTestSignal(rng, in[:])
			if block%500 == 499 {
				for k := range got.sections {
					got.sections[k].Reset()
					want.sections[k].Reset()
				}
			}
			outGot := make([]float32, BlockSize/factor)
			outWant := make([]float32, BlockSize/factor)
			got.Decimate(in[:], outGot)
			refDecimate(want, in[:], outWant)
			if i := equalBits(outGot, outWant); i >= 0 {
				t.Fatalf("factor %d, block %d: out[%d] = %v, want %v", factor, block, i, outGot[i], outWant[i])
			}
		}
	}
}

func TestHighPassFilterMatchesSectionBySection(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for _, rate := range []int{16000, 32000, 48000} {
		got := NewHighPassFilter(rate, 2)
		want := []*CascadedBiQuadFilter{
			NewCascadedBiQuadFilter(ec3ChooseHighPassCoefficients(rate)),
			NewCascadedBiQuadFilter(ec3ChooseHighPassCoefficients(rate)),
		}
		for block := 0; block < 500; block++ {
			audio := [][]float32{make([]float32, SplitBandSize), make([]float32, SplitBandSize)}
			ref := [][]float32{make([]float32, SplitBandSize), make([]float32, SplitBandSize)}
			for ch := range audio {
				biquadTestSignal(rng, audio[ch])
				copy(ref[ch], audio[ch])
				refCascade(ref[ch], ref[ch], want[ch].biquads)
			}
			got.Process(audio)
			for ch := range audio {
				if i := equalBits(audio[ch], ref[ch]); i >= 0 {
					t.Fatalf("rate %d, block %d, channel %d: y[%d] = %v, want %v", rate, block, ch, i, audio[ch][i], ref[ch][i])
				}
			}
		}
	}
}

func TestDecimatorAllocationFree(t *testing.T) {
	for _, factor := range []int{4, 8} {
		d := NewDecimator(factor)
		var in [BlockSize]float32
		out := make([]float32, BlockSize/factor)
		if a := testing.AllocsPerRun(100, func() { d.Decimate(in[:], out) }); a != 0 {
			t.Fatalf("factor %d: Decimate allocates %v times per call", factor, a)
		}
	}
}

func BenchmarkDecimate(b *testing.B) {
	var in [BlockSize]float32
	for i := range in {
		in[i] = float32(i*37%200) * 20
	}
	for _, factor := range []int{4, 8} {
		out := make([]float32, BlockSize/factor)
		b.Run(map[int]string{4: "ds4", 8: "ds8"}[factor]+"/section_by_section", func(b *testing.B) {
			d := NewDecimator(factor)
			for i := 0; i < b.N; i++ {
				refDecimate(d, in[:], out)
			}
		})
		b.Run(map[int]string{4: "ds4", 8: "ds8"}[factor]+"/sample_major", func(b *testing.B) {
			d := NewDecimator(factor)
			for i := 0; i < b.N; i++ {
				d.Decimate(in[:], out)
			}
		})
	}
}
