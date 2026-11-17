//go:build webrtc_oracle || (ooura_sse2_oracle && amd64)

package webrtcoracle

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// oouraKernOpts are the Aec3Optimization values the C++ build of this
// architecture implements.
func oouraKernOpts() []aec3.Aec3Optimization {
	if runtime.GOARCH == "amd64" {
		return []aec3.Aec3Optimization{aec3.OptNone, aec3.OptSse2, aec3.OptAvx2}
	}
	return []aec3.Aec3Optimization{aec3.OptNone, aec3.OptNeon}
}

func oouraKernVec(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		switch rng.Intn(6) {
		case 0:
			v[i] = math.Float32frombits(0x80000000)
		case 1:
			v[i] = math.Float32frombits(uint32(rng.Intn(1 << 23)))
		default:
			v[i] = float32(rng.NormFloat64() * 1000)
		}
	}
	return v
}

func oouraKernSame(t *testing.T, what string, opt aec3.Aec3Optimization, got, want []float32) {
	t.Helper()
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s opt=%d idx %d/%d: got %v (%#x) want %v (%#x)", what, opt, i, len(want),
				got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestOouraKernSpectrumBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 5000; trial++ {
		var d aec3.FftData
		copy(d.Re[:], oouraKernVec(rng, 65))
		copy(d.Im[:], oouraKernVec(rng, 65))
		for _, opt := range oouraKernOpts() {
			var got, want [65]float32
			d.Spectrum(opt, got[:])
			KernSpectrum(int(opt), &d.Re, &d.Im, &want)
			oouraKernSame(t, "Spectrum", opt, got[:], want[:])
		}
	}
}

func TestOouraKernVectorMathBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for n := 0; n <= 70; n++ {
		for trial := 0; trial < 30; trial++ {
			x, y, z := oouraKernVec(rng, n), oouraKernVec(rng, n), oouraKernVec(rng, n)
			for _, opt := range oouraKernOpts() {
				vm := aec3.NewVectorMath(opt)
				got, want := make([]float32, n), make([]float32, n)
				vm.Multiply(x, y, got)
				KernMultiply(int(opt), x, y, want)
				oouraKernSame(t, "Multiply", opt, got, want)

				copy(got, z)
				copy(want, z)
				vm.Accumulate(x, got)
				KernAccumulate(int(opt), x, want)
				oouraKernSame(t, "Accumulate", opt, got, want)

				for i := range got {
					got[i] = float32(math.Abs(float64(x[i])))
				}
				copy(want, got)
				vm.Sqrt(got)
				KernSqrt(int(opt), want)
				oouraKernSame(t, "Sqrt", opt, got, want)
			}
		}
	}
}
