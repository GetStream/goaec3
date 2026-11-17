package aec3

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

func kernRandVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		switch rng.Intn(8) {
		case 0:
			v[i] = math.Float32frombits(rng.Uint32()) // any bit pattern
		case 1:
			v[i] = math.Float32frombits(0x80000000)
		case 2:
			v[i] = math.Float32frombits(uint32(rng.Intn(1 << 23))) // subnormal
		default:
			v[i] = float32(rng.NormFloat64() * 1000)
		}
	}
	return v
}

func kernSameBits(a, b float32) bool {
	if a != a && b != b {
		return true // NaN payloads are not part of the contract
	}
	return math.Float32bits(a) == math.Float32bits(b)
}

func kernCheck(t *testing.T, name string, got, want []float32) {
	t.Helper()
	for i := range want {
		if !kernSameBits(got[i], want[i]) {
			t.Fatalf("%s: idx %d of %d: got %v (%#x) want %v (%#x)", name, i, len(want),
				got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestKernVectorKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for n := 0; n <= 70; n++ {
		for trial := 0; trial < 20; trial++ {
			x, y, z := kernRandVec(rng, n), kernRandVec(rng, n), kernRandVec(rng, n)
			got, want := make([]float32, n), make([]float32, n)
			vectorMultiply(x, y, got)
			vectorMultiplyGeneric(x, y, want)
			kernCheck(t, "Multiply", got, want)

			copy(got, z)
			copy(want, z)
			vectorAccumulate(x, got)
			vectorAccumulateGeneric(x, want)
			kernCheck(t, "Accumulate", got, want)

			for i := range x {
				x[i] = float32(math.Abs(float64(x[i])))
			}
			copy(got, x)
			copy(want, x)
			kernVectorSqrt(got)
			kernVectorSqrtGeneric(want)
			kernCheck(t, "Sqrt", got, want)
		}
	}
}

func TestKernSpectrum(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 2000; trial++ {
		var re, im, got, want [FftLengthBy2Plus1]float32
		copy(re[:], kernRandVec(rng, FftLengthBy2Plus1))
		copy(im[:], kernRandVec(rng, FftLengthBy2Plus1))
		for _, opt := range []Aec3Optimization{OptNone, OptSse2, OptAvx2, OptNeon} {
			fftDataSpectrum(opt, &re, &im, &got)
			if isAMD64 && opt == OptAvx2 {
				kernSpectrumAvx2Ref(&re, &im, &want)
			} else {
				fftDataSpectrumGeneric(&re, &im, &want)
			}
			kernCheck(t, "Spectrum", got[:], want[:])
		}
	}
}

// kernFmafExact rounds a*b+c once to float32 using exact big.Float
// arithmetic.
func kernFmafExact(a, b, c float32) float32 {
	if math.IsInf(float64(a), 0) || math.IsInf(float64(b), 0) || math.IsInf(float64(c), 0) ||
		a != a || b != b || c != c {
		return float32(float64(a)*float64(b) + float64(c))
	}
	p := new(big.Float).SetPrec(0).SetFloat64(float64(a))
	p.SetPrec(200).Mul(p, new(big.Float).SetFloat64(float64(b)))
	p.Add(p, new(big.Float).SetFloat64(float64(c)))
	if p.Sign() == 0 {
		return float32(float64(a)*float64(b) + float64(c)) // IEEE zero sign
	}
	f, _ := p.Float32()
	return f
}

func TestKernFmaf(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	// Random operands plus near-cancellation cases where double rounding
	// through float64 would go wrong.
	for i := 0; i < 300000; i++ {
		var a, b, c float32
		switch i % 3 {
		case 0:
			a, b, c = math.Float32frombits(rng.Uint32()), math.Float32frombits(rng.Uint32()), math.Float32frombits(rng.Uint32())
		case 1:
			a = float32(rng.NormFloat64())
			b = a
			c = float32(rng.NormFloat64())
		default:
			a = float32(1 + rng.Float64())
			b = float32(1 + rng.Float64())
			c = -float32(float64(a) * float64(b))
			c = math.Float32frombits(math.Float32bits(c) + uint32(rng.Intn(3)) - 1)
		}
		got, want := kernFmaf(a, b, c), kernFmafExact(a, b, c)
		if !kernSameBits(got, want) {
			t.Fatalf("fmaf(%v, %v, %v) = %v (%#x), want %v (%#x)", a, b, c,
				got, math.Float32bits(got), want, math.Float32bits(want))
		}
	}
	// An exact float32 tie: (1+2^-12)^2 - 2^-11 = 1 + 2^-24.
	a, b, c := float32(1+1.0/(1<<12)), float32(1+1.0/(1<<12)), float32(-1.0/(1<<11))
	if got, want := kernFmaf(a, b, c), kernFmafExact(a, b, c); got != want {
		t.Fatalf("fmaf tie case: %v want %v", got, want)
	}
}

func TestKernNoAllocs(t *testing.T) {
	var re, im, out [FftLengthBy2Plus1]float32
	x := make([]float32, FftLengthBy2Plus1)
	z := make([]float32, FftLengthBy2Plus1)
	n := testing.AllocsPerRun(100, func() {
		fftDataSpectrum(OptAvx2, &re, &im, &out)
		fftDataSpectrum(OptNeon, &re, &im, &out)
		vectorMultiply(x, x, z)
		vectorAccumulate(x, z)
		kernVectorSqrt(z)
	})
	if n != 0 {
		t.Fatalf("kernels allocate %v times", n)
	}
}

func BenchmarkKern(b *testing.B) {
	var re, im, out [FftLengthBy2Plus1]float32
	rng := rand.New(rand.NewSource(1))
	for i := range re {
		re[i], im[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	x, z := re[:], make([]float32, FftLengthBy2Plus1)
	opt := DetectOptimization()
	b.Run("Spectrum", func(b *testing.B) {
		for b.Loop() {
			fftDataSpectrum(opt, &re, &im, &out)
		}
	})
	b.Run("SpectrumGeneric", func(b *testing.B) {
		for b.Loop() {
			fftDataSpectrumGeneric(&re, &im, &out)
		}
	})
	b.Run("Multiply", func(b *testing.B) {
		for b.Loop() {
			vectorMultiply(x, x, z)
		}
	})
	b.Run("MultiplyGeneric", func(b *testing.B) {
		for b.Loop() {
			vectorMultiplyGeneric(x, x, z)
		}
	})
	b.Run("Accumulate", func(b *testing.B) {
		for b.Loop() {
			vectorAccumulate(x, z)
		}
	})
	b.Run("AccumulateGeneric", func(b *testing.B) {
		for b.Loop() {
			vectorAccumulateGeneric(x, z)
		}
	})
	b.Run("Sqrt", func(b *testing.B) {
		for b.Loop() {
			copy(z, x)
			kernVectorSqrt(z)
		}
	})
	b.Run("SqrtGeneric", func(b *testing.B) {
		for b.Loop() {
			copy(z, x)
			kernVectorSqrtGeneric(z)
		}
	})
}
