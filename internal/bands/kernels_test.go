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

func TestAudioUtilKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 3000; trial++ {
		n := trial % 97
		if trial%10 == 0 {
			n = 480
		}
		scale := float32(1)
		if trial%2 == 0 {
			scale = 32768
		}
		src := make([]float32, n)
		for i := range src {
			src[i] = bsRandSample(rng, scale)
		}
		if trial < 4*len(bsSpecials) {
			// Every special in every lane position.
			for i := range src {
				src[i] = bsSpecials[(i+trial)%len(bsSpecials)]
			}
		}
		off := trial % 3 // unaligned destinations

		got := make([]float32, n+off)
		want := make([]float32, n+off)
		FloatToFloatS16Slice(src, got[off:])
		floatToFloatS16SliceRef(src, want[off:])
		bsCheck(t, "FloatToFloatS16Slice", got, want)

		FloatS16ToFloatSlice(src, got[off:])
		floatS16ToFloatSliceRef(src, want[off:])
		bsCheck(t, "FloatS16ToFloatSlice", got, want)

		// In place.
		copy(got, src)
		FloatToFloatS16Slice(got[:n], got[:n])
		floatToFloatS16SliceRef(src, want)
		bsCheck(t, "FloatToFloatS16Slice in place", got[:n], want[:n])

		got16 := make([]int16, n+off)
		want16 := make([]int16, n+off)
		FloatS16ToS16Slice(src, got16[off:])
		floatS16ToS16SliceRef(src, want16[off:])
		bsCheck16(t, "FloatS16ToS16Slice", got16, want16)

		s16 := make([]int16, n)
		for i := range s16 {
			s16[i] = int16(rng.Uint32())
		}
		if n > 0 {
			s16[0], s16[n-1] = -32768, 32767
		}
		S16ToFloatS16Slice(s16, got[off:])
		s16ToFloatS16SliceRef(s16, want[off:])
		bsCheck(t, "S16ToFloatS16Slice", got, want)
	}
}

func BenchmarkAudioUtil(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	src := bsBenchVec(rng, 480, 0.3)
	srcS16 := bsBenchVec(rng, 480, 3000)
	dst := make([]float32, 480)
	dst16 := make([]int16, 480)
	for i := range dst16 {
		dst16[i] = int16(rng.Intn(65536) - 32768)
	}
	for _, c := range []struct {
		name      string
		fast, ref func()
	}{
		{"FloatToFloatS16", func() { FloatToFloatS16Slice(src, dst) }, func() { floatToFloatS16SliceRef(src, dst) }},
		{"FloatS16ToFloat", func() { FloatS16ToFloatSlice(srcS16, dst) }, func() { floatS16ToFloatSliceRef(srcS16, dst) }},
		{"FloatS16ToS16", func() { FloatS16ToS16Slice(srcS16, dst16) }, func() { floatS16ToS16SliceRef(srcS16, dst16) }},
		{"S16ToFloatS16", func() { S16ToFloatS16Slice(dst16, dst) }, func() { s16ToFloatS16SliceRef(dst16, dst) }},
	} {
		b.Run(c.name+"/dispatch", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				c.fast()
			}
		})
		b.Run(c.name+"/ref", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				c.ref()
			}
		})
	}
}

func bsRandState(rng *rand.Rand) [6]int32 {
	var s [6]int32
	for i := range s {
		switch rng.Intn(4) {
		case 0:
			s[i] = int32(rng.Uint32()) // anything, including near the int32 limits
		case 1:
			s[i] = [...]int32{-2147483648, 2147483647, 0, -1}[rng.Intn(4)]
		default:
			s[i] = int32(rng.Intn(1<<27) - 1<<26)
		}
	}
	return s
}

func bsRandS16(rng *rand.Rand, n int) []int16 {
	v := make([]int16, n)
	loud := rng.Intn(3) == 0
	for i := range v {
		switch {
		case rng.Intn(20) == 0:
			v[i] = [...]int16{-32768, 32767, 0, -1, 1}[rng.Intn(5)]
		case loud:
			v[i] = int16(rng.Uint32())
		default:
			v[i] = int16(rng.NormFloat64() * 3000)
		}
	}
	return v
}

func TestQMFKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 400; trial++ {
		bandLength := 1 + rng.Intn(bsMaxBandFrameLength)
		if trial%2 == 0 {
			bandLength = bsSamplesPerBand
		}
		a1, a2 := bsRandState(rng), bsRandState(rng)
		if trial%3 == 0 {
			a1, a2 = [6]int32{}, [6]int32{}
		}
		r1, r2 := a1, a2
		s1, s2 := bsRandState(rng), bsRandState(rng)
		q1, q2 := s1, s2
		for frame := 0; frame < 20; frame++ {
			in := bsRandS16(rng, 2*bandLength)
			gl, gh := make([]int16, bandLength), make([]int16, bandLength)
			wl, wh := make([]int16, bandLength), make([]int16, bandLength)
			AnalysisQMF(in, gl, gh, &a1, &a2)
			analysisQMFRef(in, wl, wh, &r1, &r2)
			bsCheck16(t, "AnalysisQMF low", gl, wl)
			bsCheck16(t, "AnalysisQMF high", gh, wh)
			if a1 != r1 || a2 != r2 {
				t.Fatalf("AnalysisQMF state: got %v %v want %v %v", a1, a2, r1, r2)
			}

			lo, hi := bsRandS16(rng, bandLength), bsRandS16(rng, bandLength)
			if frame%2 == 0 {
				lo, hi = wl, wh
			}
			gout, wout := make([]int16, 2*bandLength), make([]int16, 2*bandLength)
			SynthesisQMF(lo, hi, gout, &s1, &s2)
			synthesisQMFRef(lo, hi, wout, &q1, &q2)
			bsCheck16(t, "SynthesisQMF", gout, wout)
			if s1 != q1 || s2 != q2 {
				t.Fatalf("SynthesisQMF state: got %v %v want %v %v", s1, s2, q1, q2)
			}
		}
	}
}

func BenchmarkQMF(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	in := make([]int16, bsTwoBandFilterSamplesPerFrame)
	for i := range in {
		in[i] = int16(rng.NormFloat64() * 3000)
	}
	var lo, hi [bsSamplesPerBand]int16
	var out [bsTwoBandFilterSamplesPerFrame]int16
	var s1, s2 [6]int32
	b.Run("Analysis/dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AnalysisQMF(in, lo[:], hi[:], &s1, &s2)
		}
	})
	b.Run("Analysis/ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			analysisQMFRef(in, lo[:], hi[:], &s1, &s2)
		}
	})
	b.Run("Synthesis/dispatch", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			SynthesisQMF(lo[:], hi[:], out[:], &s1, &s2)
		}
	})
	b.Run("Synthesis/ref", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			synthesisQMFRef(lo[:], hi[:], out[:], &s1, &s2)
		}
	})
}
