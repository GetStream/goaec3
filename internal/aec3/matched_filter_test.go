package aec3

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

func dlyRandSlice(rng *rand.Rand, n int, scale float64) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64() * scale)
	}
	return s
}

func dlyRequireBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: got %v (%#x), want %v (%#x)", what, i, got[i], math.Float32bits(got[i]),
				want[i], math.Float32bits(want[i]))
		}
	}
}

func TestDlyFmafCorrectlyRounded(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	exact := func(a, b, c float32) float32 {
		p := new(big.Float).SetPrec(400).SetFloat64(float64(a))
		p.Mul(p, new(big.Float).SetPrec(400).SetFloat64(float64(b)))
		p.Add(p, new(big.Float).SetPrec(400).SetFloat64(float64(c)))
		f, _ := p.Float32()
		return f
	}
	for i := 0; i < 200000; i++ {
		a := float32(rng.NormFloat64())
		b := float32(rng.NormFloat64())
		var c float32
		switch i % 3 {
		case 0:
			c = float32(rng.NormFloat64())
		case 1:
			// Near-cancellation and halfway cases.
			c = -float32(float64(a)*float64(b)) * (1 + float32(rng.Intn(5)-2)*0x1p-24)
		default:
			c = float32(rng.NormFloat64() * 1e-8)
		}
		if got, want := dlyFmaf(a, b, c), exact(a, b, c); math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("dlyFmaf(%v, %v, %v) = %v, want %v", a, b, c, got, want)
		}
	}
}

// TestDlyMatchedFilterKernelsMatchReference checks that the kernels the
// build selects (archsimd or reference) are bit-identical to the pure-Go
// lane references.
func TestDlyMatchedFilterKernelsMatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 4000; trial++ {
		// Every residue modulo the unroll widths, short lengths below one
		// unrolled iteration, and x longer than h.
		n := rng.Intn(80)
		if trial%3 == 0 {
			n = rng.Intn(700)
		}
		h := dlyRandSlice(rng, n, 0.3)
		x := dlyRandSlice(rng, n+rng.Intn(20), 1000)
		y := float32(rng.NormFloat64() * 1000)

		var a8, b8 [8]float32
		for i := range a8 {
			a8[i] = float32(rng.NormFloat64())
		}
		b8 = a8
		dlyDotNeon(h, x, &a8)
		dlyDotNeonRef(h, x, &b8)
		dlyRequireBits(t, "dotNeon", a8[:], b8[:])

		var a16, b16 [16]float32
		for i := range a16 {
			a16[i] = float32(rng.NormFloat64())
		}
		b16 = a16
		dlyDotSse2(h, x, &a16)
		dlyDotSse2Ref(h, x, &b16)
		dlyRequireBits(t, "dotSse2", a16[:], b16[:])

		var a32, b32 [32]float32
		for i := range a32 {
			a32[i] = float32(rng.NormFloat64())
		}
		b32 = a32
		dlyDotAvx2(h, x, &a32)
		dlyDotAvx2Ref(h, x, &b32)
		dlyRequireBits(t, "dotAvx2", a32[:], b32[:])

		accA := dlyRandSlice(rng, n/4, 10)
		accB := append([]float32(nil), accA...)
		var x4a, x4b [4]float32
		sA := dlyAccErrNeon(h, x, y, accA, &x4a)
		sB := dlyAccErrNeonRef(h, x, y, accB, &x4b)
		dlyRequireBits(t, "accErrNeon", append(accA, append(x4a[:], sA)...), append(accB, append(x4b[:], sB)...))

		accA = dlyRandSlice(rng, n/4, 10)
		accB = append([]float32(nil), accA...)
		var x8a, x8b [8]float32
		sA = dlyAccErrSse2(h, x, y, accA, &x8a)
		sB = dlyAccErrSse2Ref(h, x, y, accB, &x8b)
		dlyRequireBits(t, "accErrSse2", append(accA, append(x8a[:], sA)...), append(accB, append(x8b[:], sB)...))

		accA = dlyRandSlice(rng, n/4, 10)
		accB = append([]float32(nil), accA...)
		var x16a, x16b [16]float32
		sA = dlyAccErrAvx2(h, x, y, accA, &x16a)
		sB = dlyAccErrAvx2Ref(h, x, y, accB, &x16b)
		dlyRequireBits(t, "accErrAvx2", append(accA, append(x16a[:], sA)...), append(accB, append(x16b[:], sB)...))

		m := rng.Intn(n + 1)
		alpha := float32(rng.NormFloat64() * 1e-4)
		hA := append([]float32(nil), h[:m]...)
		hB := append([]float32(nil), h[:m]...)
		dlyAxpy(hA, x, alpha)
		dlyAxpyRef(hB, x, alpha)
		dlyRequireBits(t, "axpy", hA, hB)

		hA = append(hA[:0], h[:m]...)
		hB = append(hB[:0], h[:m]...)
		dlyAxpyFma(hA, x, alpha)
		dlyAxpyFmaRef(hB, x, alpha)
		dlyRequireBits(t, "axpyFma", hA, hB)

		xa := dlyRandSlice(rng, n+rng.Intn(20), 1000)
		for _, g := range []int{4, 8, 16} {
			mg := n &^ (g - 1)
			hA = append(hA[:0], h[:mg]...)
			hB = append(hB[:0], h[:mg]...)
			for i := range a32 {
				a32[i] = float32(rng.NormFloat64())
			}
			b32 = a32
			switch g {
			case 4:
				dlyAxpyDotNeon(hA, xa, x, alpha, (*[8]float32)(a32[:8]))
				dlyAxpyDotNeonRef(hB, xa, x, alpha, (*[8]float32)(b32[:8]))
			case 8:
				dlyAxpyDotSse2(hA, xa, x, alpha, (*[16]float32)(a32[:16]))
				dlyAxpyDotSse2Ref(hB, xa, x, alpha, (*[16]float32)(b32[:16]))
			default:
				dlyAxpyFmaDotAvx2(hA, xa, x, alpha, &a32)
				dlyAxpyFmaDotAvx2Ref(hB, xa, x, alpha, &b32)
			}
			dlyRequireBits(t, "axpyDot h", hA, hB)
			dlyRequireBits(t, "axpyDot acc", a32[:], b32[:])
		}
	}
}

// TestDlyMatchedFilterCoreLanesDiffer guards against lanes silently
// collapsing into one another: the reduction orders must give different
// results on generic data.
func TestDlyMatchedFilterCoreLanesDiffer(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const hSize = 512
	x := dlyRandSlice(rng, 1200, 3000)
	y := dlyRandSlice(rng, 16, 3000)
	results := map[Aec3Optimization][]float32{}
	for _, opt := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		h := make([]float32, hSize)
		mf := &MatchedFilter{optimization: opt}
		var updated bool
		var errSum float32
		for rep := 0; rep < 20; rep++ {
			mf.runCore(1190-rep, 1, 0.7, x, y, h, &updated, &errSum, false)
		}
		results[opt] = append(h, errSum)
	}
	for _, a := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		for _, b := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
			if a >= b {
				continue
			}
			same := true
			for i := range results[a] {
				if math.Float32bits(results[a][i]) != math.Float32bits(results[b][i]) {
					same = false
					break
				}
			}
			if same {
				t.Errorf("lanes %d and %d produced identical results", a, b)
			}
		}
	}
}

func BenchmarkMatchedFilterCore(b *testing.B) {
	rng := rand.New(rand.NewSource(4))
	x := dlyRandSlice(rng, GetDownSampledBufferSize(4, 5), 3000)
	y := dlyRandSlice(rng, 16, 3000)
	for _, tc := range []struct {
		name     string
		opt      Aec3Optimization
		preEcho  bool
		startIdx int
	}{
		{"None", OptNone, false, 100}, {"Neon", OptNeon, false, 100}, {"Sse2", OptSse2, false, 100},
		{"Avx2", OptAvx2, false, 100}, {"NeonAccErr", OptNeon, true, 100}, {"Sse2AccErr", OptSse2, true, 100},
		{"Avx2AccErr", OptAvx2, true, 100},
	} {
		b.Run(tc.name, func(b *testing.B) {
			mf := NewMatchedFilter(tc.opt, 16, 32, 5, 24, 150, 0.7, 0.7, 0.2, true)
			h := dlyRandSlice(rng, 512, 0.01)
			var updated bool
			var errSum float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				mf.runCore(tc.startIdx, 1e30, 0.7, x, y, h, &updated, &errSum, tc.preEcho)
			}
		})
	}
}
