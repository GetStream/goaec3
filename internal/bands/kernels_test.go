package bands

import (
	"math"
	"math/rand"
	"testing"
)

// The kernel tests compare the dispatched implementations (archsimd under
// GOEXPERIMENT=simd, the references otherwise) with the scalar references
// bit for bit.

func bsCheck(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: got %v (%#x) want %v (%#x)", what, i, got[i], math.Float32bits(got[i]),
				want[i], math.Float32bits(want[i]))
		}
	}
}

func bsCheck16(t *testing.T, what string, got, want []int16) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: got %d want %d", what, i, got[i], want[i])
		}
	}
}

// bsSpecials are values that exercise signed zeros, clamping boundaries,
// rounding ties, infinities and NaNs.
var bsSpecials = []float32{
	0, float32(math.Copysign(0, -1)), 1, -1, 0.5, -0.5, 1.5, -1.5, 2, -2,
	32767, -32767, 32768, -32768, 32767.5, -32768.5, 32766.5, -32767.5, 40000, -40000,
	1e-40, -1e-40, math.MaxFloat32, -math.MaxFloat32, 1.0000001, -1.0000001, 0.99999994, -0.99999994,
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	math.Float32frombits(0x7fa00001), math.Float32frombits(0xffc00123),
}

// bsRandSample returns a FloatS16-scale sample: mostly ordinary values,
// sometimes overdriven, small, exactly integral or special.
func bsRandSample(rng *rand.Rand, scale float32) float32 {
	switch r := rng.Intn(40); {
	case r == 0:
		return bsSpecials[rng.Intn(len(bsSpecials))]
	case r == 1:
		return float32(rng.Intn(70000) - 35000)
	case r == 2:
		return float32(rng.Intn(140001)-70000) / 2
	case r == 3:
		return float32(rng.NormFloat64()) * 1e-3
	case r < 6:
		return float32(rng.NormFloat64()) * 2 * scale
	default:
		return float32(rng.NormFloat64()) * 0.3 * scale
	}
}

// bsBenchVec returns finite Gaussian samples for benchmarks.
func bsBenchVec(rng *rand.Rand, n int, scale float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64()) * scale
	}
	return v
}

func TestThreeBandFilterBankKernel(t *testing.T) {
	for seed := int64(1); seed <= 20; seed++ {
		rng := rand.New(rand.NewSource(seed))
		var fast ThreeBandFilterBank
		var ref threeBandFilterBankRef
		var fb, rb [NumBands][SplitBandSize]float32
		for frame := 0; frame < 200; frame++ {
			var in [FullBandSize]float32
			finite := seed%4 != 0
			for i := range in {
				in[i] = bsRandSample(rng, 32768)
				if finite && (math.IsNaN(float64(in[i])) || math.IsInf(float64(in[i]), 0)) {
					in[i] = 0
				}
			}
			fo := [NumBands][]float32{fb[0][:], fb[1][:], fb[2][:]}
			ro := [NumBands][]float32{rb[0][:], rb[1][:], rb[2][:]}
			fast.Analysis(&in, &fo)
			ref.analysis(&in, &ro)
			for b := range fb {
				bsCheck(t, "analysis", fb[b][:], rb[b][:])
			}

			// Synthesis on the analysis output and on independent bands.
			if frame%2 == 1 {
				for b := range rb {
					for i := range rb[b] {
						rb[b][i] = bsRandSample(rng, 32768)
						if finite && (math.IsNaN(float64(rb[b][i])) || math.IsInf(float64(rb[b][i]), 0)) {
							rb[b][i] = 0
						}
					}
				}
			}
			var fout, rout [FullBandSize]float32
			fout[7], rout[7] = 1, 2 // outputs are overwritten
			fi := [NumBands][]float32{rb[0][:], rb[1][:], rb[2][:]}
			fast.Synthesis(&fi, &fout)
			ref.synthesis(&fi, &rout)
			bsCheck(t, "synthesis", fout[:], rout[:])
		}
	}
}

func BenchmarkThreeBandFilterBankAnalysis(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var in [FullBandSize]float32
	copy(in[:], bsBenchVec(rng, FullBandSize, 3000))
	var bands [NumBands][SplitBandSize]float32
	out := [NumBands][]float32{bands[0][:], bands[1][:], bands[2][:]}
	b.Run("dispatch", func(b *testing.B) {
		var f ThreeBandFilterBank
		for i := 0; i < b.N; i++ {
			f.Analysis(&in, &out)
		}
	})
	b.Run("ref", func(b *testing.B) {
		var f threeBandFilterBankRef
		for i := 0; i < b.N; i++ {
			f.analysis(&in, &out)
		}
	})
}

func BenchmarkThreeBandFilterBankSynthesis(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	var bands [NumBands][SplitBandSize]float32
	for i := range bands {
		copy(bands[i][:], bsBenchVec(rng, SplitBandSize, 3000))
	}
	in := [NumBands][]float32{bands[0][:], bands[1][:], bands[2][:]}
	var out [FullBandSize]float32
	b.Run("dispatch", func(b *testing.B) {
		var f ThreeBandFilterBank
		for i := 0; i < b.N; i++ {
			f.Synthesis(&in, &out)
		}
	})
	b.Run("ref", func(b *testing.B) {
		var f threeBandFilterBankRef
		for i := 0; i < b.N; i++ {
			f.synthesis(&in, &out)
		}
	})
}
