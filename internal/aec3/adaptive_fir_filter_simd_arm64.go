//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// NEON kernels for adaptive_fir_filter.cc / adaptive_fir_filter_erl.cc. Each
// kernel processes bins 0..63 of one FftData with the operation order of the
// libwebrtc NEON intrinsics. vmlaq_f32/vmlsq_f32 are separate multiply and
// add/sub in the -ffp-contract=off build, so they map to Mul followed by
// Add/Sub (never MulAdd). The x86 lanes run their pure-Go references.

// afirApplyBlockNEON: S.re += X.re*H.re - X.im*H.im; S.im += X.re*H.im + X.im*H.re.
func afirApplyBlockNEON(x, h, s *FftData) {
	for k := 0; k < FftLengthBy2; k += 4 {
		xRe := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k : k+4]))
		xIm := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k : k+4]))
		hRe := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[k : k+4]))
		hIm := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[k : k+4]))
		sRe := archsimd.LoadFloat32x4Array((*[4]float32)(s.Re[k : k+4]))
		sIm := archsimd.LoadFloat32x4Array((*[4]float32)(s.Im[k : k+4]))
		e := xRe.Mul(hRe).Sub(xIm.Mul(hIm)) // vmlsq_f32(vmulq_f32(X_re, H_re), X_im, H_im)
		f := xRe.Mul(hIm).Add(xIm.Mul(hRe)) // vmlaq_f32(vmulq_f32(X_re, H_im), X_im, H_re)
		sRe.Add(e).StoreArray((*[4]float32)(s.Re[k : k+4]))
		sIm.Add(f).StoreArray((*[4]float32)(s.Im[k : k+4]))
	}
}

// afirAdaptBlockNEON: H.re += X.re*G.re + X.im*G.im; H.im += X.re*G.im - X.im*G.re.
func afirAdaptBlockNEON(x, g, h *FftData) {
	for k := 0; k < FftLengthBy2; k += 4 {
		gRe := archsimd.LoadFloat32x4Array((*[4]float32)(g.Re[k : k+4]))
		gIm := archsimd.LoadFloat32x4Array((*[4]float32)(g.Im[k : k+4]))
		xRe := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k : k+4]))
		xIm := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k : k+4]))
		hRe := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[k : k+4]))
		hIm := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[k : k+4]))
		e := xRe.Mul(gRe).Add(xIm.Mul(gIm)) // vmlaq_f32(vmulq_f32(X_re, G_re), X_im, G_im)
		f := xRe.Mul(gIm).Sub(xIm.Mul(gRe)) // vmlsq_f32(vmulq_f32(X_re, G_im), X_im, G_re)
		hRe.Add(e).StoreArray((*[4]float32)(h.Re[k : k+4]))
		hIm.Add(f).StoreArray((*[4]float32)(h.Im[k : k+4]))
	}
}

// afirFreqRespBlockNEON: H2 = max(H2, re*re + im*im).
func afirFreqRespBlockNEON(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j += 4 {
		re := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[j : j+4]))
		im := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[j : j+4]))
		p := (*[4]float32)(h2[j : j+4])
		n := re.Mul(re).Add(im.Mul(im)) // vmlaq_f32(vmulq_f32(re, re), im, im)
		archsimd.LoadFloat32x4Array(p).Max(n).StoreArray(p)
	}
}

// afirErlBlockNEON: erl += H2 over bins 0..63.
func afirErlBlockNEON(h2, erl *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[4]float32)(erl[k : k+4])
		archsimd.LoadFloat32x4Array(p).Add(archsimd.LoadFloat32x4Array((*[4]float32)(h2[k : k+4]))).StoreArray(p)
	}
}

func afirApplyBlock(opt Aec3Optimization, x, h, s *FftData) {
	if opt == OptNeon {
		afirApplyBlockNEON(x, h, s)
		return
	}
	afirApplyBinsGo(x, h, s, 0, FftLengthBy2)
}

func afirAdaptBlock(opt Aec3Optimization, x, g, h *FftData) {
	if opt == OptNeon {
		afirAdaptBlockNEON(x, g, h)
		return
	}
	afirAdaptBinsGo(x, g, h, 0, FftLengthBy2)
}

func afirFreqRespBlock(opt Aec3Optimization, h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	switch opt {
	case OptNeon:
		afirFreqRespBlockNEON(h, h2)
	case OptAvx2:
		afirFreqRespBlockAvx2Go(h, h2)
	default:
		afirFreqRespBinsGo(h, h2, 0, FftLengthBy2)
	}
}

func afirErlBlock(opt Aec3Optimization, h2, erl *[FftLengthBy2Plus1]float32) {
	if opt == OptNeon {
		afirErlBlockNEON(h2, erl)
		return
	}
	afirErlBlockGo(h2, erl)
}
