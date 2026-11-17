//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// SSE2 and AVX2 kernels for adaptive_fir_filter{,_avx2}.cc and
// adaptive_fir_filter_erl{,_avx2}.cc. Each kernel processes bins 0..63 of one
// FftData with the operation order of the libwebrtc intrinsics. Only the
// AVX2 frequency response is fused (_mm256_fmadd_ps -> MulAdd).
//
// archsimd's 128-bit float ops are VEX encoded (CPU feature AVX); they round
// exactly like the SSE2 instructions, so the SSE2 lane uses them when AVX is
// present and its pure-Go reference otherwise. The NEON lane runs its pure-Go
// reference.

var (
	afirHasAVX     = archsimd.X86.AVX()
	afirHasAVX2    = archsimd.X86.AVX2()
	afirHasAVX2FMA = afirHasAVX2 && archsimd.X86.FMA()
)

func afirApplyBlockSSE2(x, h, s *FftData) {
	for k := 0; k < FftLengthBy2; k += 4 {
		xRe := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k : k+4]))
		xIm := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k : k+4]))
		hRe := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[k : k+4]))
		hIm := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[k : k+4]))
		sRe := archsimd.LoadFloat32x4Array((*[4]float32)(s.Re[k : k+4]))
		sIm := archsimd.LoadFloat32x4Array((*[4]float32)(s.Im[k : k+4]))
		e := xRe.Mul(hRe).Sub(xIm.Mul(hIm))
		f := xRe.Mul(hIm).Add(xIm.Mul(hRe))
		sRe.Add(e).StoreArray((*[4]float32)(s.Re[k : k+4]))
		sIm.Add(f).StoreArray((*[4]float32)(s.Im[k : k+4]))
	}
}

func afirAdaptBlockSSE2(x, g, h *FftData) {
	for k := 0; k < FftLengthBy2; k += 4 {
		gRe := archsimd.LoadFloat32x4Array((*[4]float32)(g.Re[k : k+4]))
		gIm := archsimd.LoadFloat32x4Array((*[4]float32)(g.Im[k : k+4]))
		xRe := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k : k+4]))
		xIm := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k : k+4]))
		hRe := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[k : k+4]))
		hIm := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[k : k+4]))
		e := xRe.Mul(gRe).Add(xIm.Mul(gIm))
		f := xRe.Mul(gIm).Sub(xIm.Mul(gRe))
		hRe.Add(e).StoreArray((*[4]float32)(h.Re[k : k+4]))
		hIm.Add(f).StoreArray((*[4]float32)(h.Im[k : k+4]))
	}
}

func afirFreqRespBlockSSE2(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j += 4 {
		re := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[j : j+4]))
		im := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[j : j+4]))
		p := (*[4]float32)(h2[j : j+4])
		n := re.Mul(re).Add(im.Mul(im))
		archsimd.LoadFloat32x4Array(p).Max(n).StoreArray(p)
	}
}

func afirErlBlockSSE2(h2, erl *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[4]float32)(erl[k : k+4])
		archsimd.LoadFloat32x4Array(p).Add(archsimd.LoadFloat32x4Array((*[4]float32)(h2[k : k+4]))).StoreArray(p)
	}
}

func afirApplyBlockAVX2(x, h, s *FftData) {
	for k := 0; k < FftLengthBy2; k += 8 {
		xRe := archsimd.LoadFloat32x8Array((*[8]float32)(x.Re[k : k+8]))
		xIm := archsimd.LoadFloat32x8Array((*[8]float32)(x.Im[k : k+8]))
		hRe := archsimd.LoadFloat32x8Array((*[8]float32)(h.Re[k : k+8]))
		hIm := archsimd.LoadFloat32x8Array((*[8]float32)(h.Im[k : k+8]))
		sRe := archsimd.LoadFloat32x8Array((*[8]float32)(s.Re[k : k+8]))
		sIm := archsimd.LoadFloat32x8Array((*[8]float32)(s.Im[k : k+8]))
		e := xRe.Mul(hRe).Sub(xIm.Mul(hIm))
		f := xRe.Mul(hIm).Add(xIm.Mul(hRe))
		sRe.Add(e).StoreArray((*[8]float32)(s.Re[k : k+8]))
		sIm.Add(f).StoreArray((*[8]float32)(s.Im[k : k+8]))
	}
}

func afirAdaptBlockAVX2(x, g, h *FftData) {
	for k := 0; k < FftLengthBy2; k += 8 {
		gRe := archsimd.LoadFloat32x8Array((*[8]float32)(g.Re[k : k+8]))
		gIm := archsimd.LoadFloat32x8Array((*[8]float32)(g.Im[k : k+8]))
		xRe := archsimd.LoadFloat32x8Array((*[8]float32)(x.Re[k : k+8]))
		xIm := archsimd.LoadFloat32x8Array((*[8]float32)(x.Im[k : k+8]))
		hRe := archsimd.LoadFloat32x8Array((*[8]float32)(h.Re[k : k+8]))
		hIm := archsimd.LoadFloat32x8Array((*[8]float32)(h.Im[k : k+8]))
		e := xRe.Mul(gRe).Add(xIm.Mul(gIm))
		f := xRe.Mul(gIm).Sub(xIm.Mul(gRe))
		hRe.Add(e).StoreArray((*[8]float32)(h.Re[k : k+8]))
		hIm.Add(f).StoreArray((*[8]float32)(h.Im[k : k+8]))
	}
}

// afirFreqRespBlockAVX2: re2 = re*re; re2 = _mm256_fmadd_ps(im, im, re2).
func afirFreqRespBlockAVX2(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j += 8 {
		re := archsimd.LoadFloat32x8Array((*[8]float32)(h.Re[j : j+8]))
		im := archsimd.LoadFloat32x8Array((*[8]float32)(h.Im[j : j+8]))
		p := (*[8]float32)(h2[j : j+8])
		n := im.MulAdd(im, re.Mul(re))
		archsimd.LoadFloat32x8Array(p).Max(n).StoreArray(p)
	}
}

func afirErlBlockAVX2(h2, erl *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k += 8 {
		p := (*[8]float32)(erl[k : k+8])
		archsimd.LoadFloat32x8Array(p).Add(archsimd.LoadFloat32x8Array((*[8]float32)(h2[k : k+8]))).StoreArray(p)
	}
}

func afirApplyBlock(opt Aec3Optimization, x, h, s *FftData) {
	switch {
	case opt == OptSse2 && afirHasAVX:
		afirApplyBlockSSE2(x, h, s)
	case opt == OptAvx2 && afirHasAVX2:
		afirApplyBlockAVX2(x, h, s)
	default:
		afirApplyBinsGo(x, h, s, 0, FftLengthBy2)
	}
}

func afirAdaptBlock(opt Aec3Optimization, x, g, h *FftData) {
	switch {
	case opt == OptSse2 && afirHasAVX:
		afirAdaptBlockSSE2(x, g, h)
	case opt == OptAvx2 && afirHasAVX2:
		afirAdaptBlockAVX2(x, g, h)
	default:
		afirAdaptBinsGo(x, g, h, 0, FftLengthBy2)
	}
}

func afirFreqRespBlock(opt Aec3Optimization, h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	switch {
	case opt == OptSse2 && afirHasAVX:
		afirFreqRespBlockSSE2(h, h2)
	case opt == OptAvx2 && afirHasAVX2FMA:
		afirFreqRespBlockAVX2(h, h2)
	case opt == OptAvx2:
		afirFreqRespBlockAvx2Go(h, h2)
	default:
		afirFreqRespBinsGo(h, h2, 0, FftLengthBy2)
	}
}

func afirErlBlock(opt Aec3Optimization, h2, erl *[FftLengthBy2Plus1]float32) {
	switch {
	case opt == OptSse2 && afirHasAVX:
		afirErlBlockSSE2(h2, erl)
	case opt == OptAvx2 && afirHasAVX2:
		afirErlBlockAVX2(h2, erl)
	default:
		afirErlBlockGo(h2, erl)
	}
}
