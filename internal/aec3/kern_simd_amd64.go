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
	case kernHasAVX:
		kernSpectrumSse2x4(re, im, out)
	default:
		fftDataSpectrumGeneric(re, im, out)
	}
}

// kernSpectrumAvx2 is FftData::SpectrumAVX2: fma(r, r, i*i) for bins 0..63.
func kernSpectrumAvx2(re, im, out *[FftLengthBy2Plus1]float32) {
	var zero archsimd.Float32x8
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		r := archsimd.LoadFloat32x8Array((*[8]float32)(re[k : k+8]))
		i := archsimd.LoadFloat32x8Array((*[8]float32)(im[k : k+8]))
		r.MulAdd(r, i.Mul(i)).StoreArray((*[8]float32)(out[k : k+8]))
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
	const n = FftLengthBy2
	out[n] = float32(re[n]*re[n]) + float32(im[n]*im[n])
}

// kernSpectrumSse2 is the unfused r*r + i*i of the scalar/SSE2 lanes.
func kernSpectrumSse2(re, im, out *[FftLengthBy2Plus1]float32) {
	var zero archsimd.Float32x8
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		r := archsimd.LoadFloat32x8Array((*[8]float32)(re[k : k+8]))
		i := archsimd.LoadFloat32x8Array((*[8]float32)(im[k : k+8]))
		r.Mul(r).Add(i.Mul(i)).StoreArray((*[8]float32)(out[k : k+8]))
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
	const n = FftLengthBy2
	out[n] = float32(re[n]*re[n]) + float32(im[n]*im[n])
}

// kernSpectrumSse2x4 is kernSpectrumSse2 with 128-bit operations, for
// CPUs (or GODEBUG settings) with AVX but not AVX2.
func kernSpectrumSse2x4(re, im, out *[FftLengthBy2Plus1]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		r := archsimd.LoadFloat32x4Array((*[4]float32)(re[k : k+4]))
		i := archsimd.LoadFloat32x4Array((*[4]float32)(im[k : k+4]))
		r.Mul(r).Add(i.Mul(i)).StoreArray((*[4]float32)(out[k : k+4]))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
	const n = FftLengthBy2
	out[n] = float32(re[n]*re[n]) + float32(im[n]*im[n])
}

func vectorMultiply(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	if kernHasAVX && kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] * y[64]
		}
		if kernHasAVX2 {
			kernMul64y((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		} else {
			kernMul64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		}
		return
	}
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
	if kernHasAVX && kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] += x[64]
		}
		if kernHasAVX2 {
			kernAdd64y((*[64]float32)(z), (*[64]float32)(x), (*[64]float32)(z))
		} else {
			kernAdd64((*[64]float32)(z), (*[64]float32)(x), (*[64]float32)(z))
		}
		return
	}
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
	if kernHasAVX && kernIs64(len(x)) {
		if len(x) == 65 {
			kernVectorSqrtGeneric(x[64:])
		}
		if kernHasAVX2 {
			kernSqrt64y((*[64]float32)(x))
		} else {
			kernSqrt64((*[64]float32)(x))
		}
		return
	}
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

// kernSplat4 returns x in all four lanes with AVX instructions (VMOVSS,
// VSHUFPS). archsimd.BroadcastFloat32x4 compiles to VBROADCASTSS from a
// register, and BroadcastInt32x4 to VPBROADCASTD, which are AVX2
// instructions: the 128-bit kernels run on CPUs with AVX only.
func kernSplat4(x float32) archsimd.Float32x4 {
	var z archsimd.Float32x4
	v := z.SetElem(0, x)
	return v.ConcatPermuteScalars(0, 0, 0, 0, v)
}

// kernSplatI4 is kernSplat4 for int32 lanes (VMOVD, VPSHUFD).
func kernSplatI4(x int32) archsimd.Int32x4 {
	var z archsimd.Int32x4
	return z.SetElem(0, x).PermuteScalars(0, 0, 0, 0)
}
