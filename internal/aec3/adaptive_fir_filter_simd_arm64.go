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

// afirHasTermKernels reports that afirApplyTerms/afirAdaptTerms are archsimd
// kernels. Every lane computes the same rounded products and per-bin
// accumulation order, so one kernel serves the NEON, SSE2 and AVX2 lanes.
const afirHasTermKernels = true

// afirApplyTerms is afirApplyTermsGo with the bin loop outside the pair loop:
// eight bins of S stay in registers while all pairs accumulate into them, in
// pair order, starting from +0.
func afirApplyTerms(xs, hs []*FftData, s *FftData) {
	hs = hs[:len(xs)]
	for k := 0; k < FftLengthBy2; k += 8 {
		var re0, re1, im0, im1 archsimd.Float32x4
		for t, x := range xs {
			h := hs[t]
			xr0, xr1 := afirLd8(&x.Re, k)
			xi0, xi1 := afirLd8(&x.Im, k)
			hr0, hr1 := afirLd8(&h.Re, k)
			hi0, hi1 := afirLd8(&h.Im, k)
			re0 = re0.Add(xr0.Mul(hr0).Sub(xi0.Mul(hi0)))
			re1 = re1.Add(xr1.Mul(hr1).Sub(xi1.Mul(hi1)))
			im0 = im0.Add(xr0.Mul(hi0).Add(xi0.Mul(hr0)))
			im1 = im1.Add(xr1.Mul(hi1).Add(xi1.Mul(hr1)))
		}
		afirSt8(&s.Re, k, re0, re1)
		afirSt8(&s.Im, k, im0, im1)
	}
	const n = FftLengthBy2
	var re, im float32
	for t, x := range xs {
		h := hs[t]
		re += float32(x.Re[n]*h.Re[n]) - float32(x.Im[n]*h.Im[n])
		im += float32(x.Re[n]*h.Im[n]) + float32(x.Im[n]*h.Re[n])
	}
	s.Re[n], s.Im[n] = re, im
}

// afirAdaptTerms is afirAdaptTermsGo with eight bins of G held in registers
// across all pairs.
func afirAdaptTerms(xs, hs []*FftData, g *FftData) {
	hs = hs[:len(xs)]
	for k := 0; k < FftLengthBy2; k += 8 {
		gr0, gr1 := afirLd8(&g.Re, k)
		gi0, gi1 := afirLd8(&g.Im, k)
		for t, x := range xs {
			h := hs[t]
			xr0, xr1 := afirLd8(&x.Re, k)
			xi0, xi1 := afirLd8(&x.Im, k)
			hr0, hr1 := afirLd8(&h.Re, k)
			hi0, hi1 := afirLd8(&h.Im, k)
			afirSt8(&h.Re, k, hr0.Add(xr0.Mul(gr0).Add(xi0.Mul(gi0))), hr1.Add(xr1.Mul(gr1).Add(xi1.Mul(gi1))))
			afirSt8(&h.Im, k, hi0.Add(xr0.Mul(gi0).Sub(xi0.Mul(gr0))), hi1.Add(xr1.Mul(gi1).Sub(xi1.Mul(gr1))))
		}
	}
	for t, x := range xs {
		afirAdaptBinsGo(x, g, hs[t], FftLengthBy2, FftLengthBy2Plus1)
	}
}

// afirLd8 loads bins k..k+7 as two vectors.
func afirLd8(a *[FftLengthBy2Plus1]float32, k int) (archsimd.Float32x4, archsimd.Float32x4) {
	p := (*[8]float32)(a[k : k+8])
	return archsimd.LoadFloat32x4Array((*[4]float32)(p[:4])), archsimd.LoadFloat32x4Array((*[4]float32)(p[4:]))
}

// afirSt8 stores two vectors to bins k..k+7.
func afirSt8(a *[FftLengthBy2Plus1]float32, k int, lo, hi archsimd.Float32x4) {
	p := (*[8]float32)(a[k : k+8])
	lo.StoreArray((*[4]float32)(p[:4]))
	hi.StoreArray((*[4]float32)(p[4:]))
}

// afirHasNeonFreqResp reports that afirFreqRespPartitionNEON is available.
const afirHasNeonFreqResp = true

// afirFreqRespPartitionNEON computes one partition of
// ComputeFrequencyResponse_Neon: H2 = +0, then H2 = vmaxq(H2, |H[ch]|^2) for
// every channel in order, with eight bins held in registers across the
// channels instead of being stored and reloaded.
func afirFreqRespPartitionNEON(hs []FftData, h2 *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k += 8 {
		var a0, a1 archsimd.Float32x4
		for c := range hs {
			h := &hs[c]
			r0, r1 := afirLd8(&h.Re, k)
			i0, i1 := afirLd8(&h.Im, k)
			a0 = a0.Max(r0.Mul(r0).Add(i0.Mul(i0)))
			a1 = a1.Max(r1.Mul(r1).Add(i1.Mul(i1)))
		}
		afirSt8(h2, k, a0, a1)
	}
	const n = FftLengthBy2
	var m float32
	for c := range hs {
		h := &hs[c]
		tmp := float32(h.Re[n]*h.Re[n]) + float32(h.Im[n]*h.Im[n])
		if m < tmp {
			m = tmp
		}
	}
	h2[n] = m
}
