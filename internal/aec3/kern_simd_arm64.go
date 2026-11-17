//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// archsimd kernels for arm64. Each element is one IEEE operation (or the
// unfused r*r + i*i), so they are bit-identical to the generic code in every
// lane; the arm64 build of libwebrtc has no fused Spectrum variant.

func fftDataSpectrum(_ Aec3Optimization, re, im, out *[FftLengthBy2Plus1]float32) {
	kernSpectrumNeon(re, im, out)
}

// kernSpectrumNeon computes re*re + im*im with separate Mul and Add.
func kernSpectrumNeon(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := 0; k+4 <= FftLengthBy2; k += 4 {
		r := archsimd.LoadFloat32x4Array((*[4]float32)(re[k : k+4]))
		i := archsimd.LoadFloat32x4Array((*[4]float32)(im[k : k+4]))
		r.Mul(r).Add(i.Mul(i)).StoreArray((*[4]float32)(out[k : k+4]))
	}
	const n = FftLengthBy2
	out[n] = float32(re[n]*re[n]) + float32(im[n]*im[n])
}

func vectorMultiply(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	j := 0
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(x[j:]).Mul(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = x[j] * y[j]
	}
}

func vectorAccumulate(x, z []float32) {
	z = z[:len(x)]
	j := 0
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(z[j:]).Add(archsimd.LoadFloat32x4(x[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] += x[j]
	}
}

// kernVectorSqrt is VectorMath::Sqrt (vsqrtq_f32 on arm64 is IEEE sqrt).
func kernVectorSqrt(x []float32) {
	j := 0
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(x[j:]).Sqrt().Store(x[j:])
	}
	kernVectorSqrtGeneric(x[j:])
}
