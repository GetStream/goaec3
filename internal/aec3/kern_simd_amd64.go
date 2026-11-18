//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// archsimd kernels for amd64. Multiply/Accumulate/Sqrt and the SSE2
// Spectrum are one IEEE operation per element, so 256-bit AVX2 or 128-bit
// AVX code gives the same bits as the SSE2/AVX2 C++. The kAvx2 Spectrum is
// fused (_mm256_fmadd_ps) and uses Float32x8.MulAdd.

var (
	kernHasAVX  = archsimd.X86.AVX()
	kernHasAVX2 = archsimd.X86.AVX2()
	kernHasFMA  = archsimd.X86.AVX2() && archsimd.X86.FMA()
)

func fftDataSpectrum(opt Aec3Optimization, re, im, out *[FftLengthBy2Plus1]float32) {
	switch {
	case opt == OptAvx2 && kernHasFMA:
		kernSpectrumAvx2(re, im, out)
	case opt == OptAvx2:
		kernSpectrumAvx2Ref(re, im, out)
	case kernHasAVX2:
		kernSpectrumSse2(re, im, out)
	default:
		fftDataSpectrumGeneric(re, im, out)
	}
}

// kernSpectrumAvx2 is FftData::SpectrumAVX2: fma(r, r, i*i) for bins 0..63.
func kernSpectrumAvx2(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := 0; k+8 <= FftLengthBy2; k += 8 {
		r := archsimd.LoadFloat32x8Array((*[8]float32)(re[k : k+8]))
		i := archsimd.LoadFloat32x8Array((*[8]float32)(im[k : k+8]))
		r.MulAdd(r, i.Mul(i)).StoreArray((*[8]float32)(out[k : k+8]))
	}
	archsimd.ClearAVXUpperBits()
	const n = FftLengthBy2
	out[n] = float32(re[n]*re[n]) + float32(im[n]*im[n])
}

// kernSpectrumSse2 is the unfused r*r + i*i of the scalar/SSE2 lanes.
func kernSpectrumSse2(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := 0; k+8 <= FftLengthBy2; k += 8 {
		r := archsimd.LoadFloat32x8Array((*[8]float32)(re[k : k+8]))
		i := archsimd.LoadFloat32x8Array((*[8]float32)(im[k : k+8]))
		r.Mul(r).Add(i.Mul(i)).StoreArray((*[8]float32)(out[k : k+8]))
	}
	archsimd.ClearAVXUpperBits()
	const n = FftLengthBy2
	out[n] = float32(re[n]*re[n]) + float32(im[n]*im[n])
}

func vectorMultiply(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	j := 0
	if kernHasAVX2 {
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(x[j:]).Mul(archsimd.LoadFloat32x8(y[j:])).Store(z[j:])
		}
		archsimd.ClearAVXUpperBits()
	}
	if kernHasAVX {
		for ; j+4 <= len(x); j += 4 {
			archsimd.LoadFloat32x4(x[j:]).Mul(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
		}
	}
	for ; j < len(x); j++ {
		z[j] = x[j] * y[j]
	}
}

func vectorAccumulate(x, z []float32) {
	z = z[:len(x)]
	j := 0
	if kernHasAVX2 {
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(z[j:]).Add(archsimd.LoadFloat32x8(x[j:])).Store(z[j:])
		}
		archsimd.ClearAVXUpperBits()
	}
	if kernHasAVX {
		for ; j+4 <= len(x); j += 4 {
			archsimd.LoadFloat32x4(z[j:]).Add(archsimd.LoadFloat32x4(x[j:])).Store(z[j:])
		}
	}
	for ; j < len(x); j++ {
		z[j] += x[j]
	}
}

// kernVectorSqrt is VectorMath::Sqrt (sqrtps is IEEE sqrt).
func kernVectorSqrt(x []float32) {
	j := 0
	if kernHasAVX2 {
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(x[j:]).Sqrt().Store(x[j:])
		}
		archsimd.ClearAVXUpperBits()
	}
	if kernHasAVX {
		for ; j+4 <= len(x); j += 4 {
			archsimd.LoadFloat32x4(x[j:]).Sqrt().Store(x[j:])
		}
	}
	kernVectorSqrtGeneric(x[j:])
}
