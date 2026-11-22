//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// NEON kernels of adaptive_fir_filter.cc and adaptive_fir_filter_erl.cc. A
// kernel processes bins 0..63 of one FftData in the operation order of the
// libwebrtc intrinsics. vmlaq_f32 and vmlsq_f32 are unfused: Mul followed by
// Add or Sub, never MulAdd. The x86 lanes run their pure-Go references.

// afirApplyBlockNEON adds x*h to s.
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

// afirAdaptBlockNEON adds conj(x)*g to h.
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

// afirFreqRespBlockNEON sets h2 = max(h2, re*re + im*im).
func afirFreqRespBlockNEON(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j += 4 {
		re := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[j : j+4]))
		im := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[j : j+4]))
		p := (*[4]float32)(h2[j : j+4])
		n := re.Mul(re).Add(im.Mul(im)) // vmlaq_f32(vmulq_f32(re, re), im, im)
		archsimd.LoadFloat32x4Array(p).Max(n).StoreArray(p)
	}
}

// afirErlBlockNEON adds h2 to erl.
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
	case OptSse2:
		afirFreqRespBlockSse2Go(h, h2)
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

// afirHasTermKernels reports whether afirApplyTerms and afirAdaptTerms replace
// the per-partition loops. The lanes agree bin for bin, so one kernel serves
// the NEON, SSE2 and AVX2 lanes.
const afirHasTermKernels = true

// afirApplyTerms is afirApplyTermsGo with the bin loop outside the pair loop:
// sixteen bins of s stay in registers while the pairs accumulate into them in
// pair order from +0.
func afirApplyTerms(xs, hs []*FftData, s *FftData) {
	hs = hs[:len(xs)]
	for k := 0; k < FftLengthBy2; k += 16 {
		var re0, re1, re2, re3, im0, im1, im2, im3 archsimd.Float32x4
		for t, x := range xs {
			h := hs[t]
			xr0, xr1, xr2, xr3 := afirLd16(&x.Re, k)
			xi0, xi1, xi2, xi3 := afirLd16(&x.Im, k)
			hr0, hr1, hr2, hr3 := afirLd16(&h.Re, k)
			hi0, hi1, hi2, hi3 := afirLd16(&h.Im, k)
			re0 = re0.Add(xr0.Mul(hr0).Sub(xi0.Mul(hi0)))
			re1 = re1.Add(xr1.Mul(hr1).Sub(xi1.Mul(hi1)))
			re2 = re2.Add(xr2.Mul(hr2).Sub(xi2.Mul(hi2)))
			re3 = re3.Add(xr3.Mul(hr3).Sub(xi3.Mul(hi3)))
			im0 = im0.Add(xr0.Mul(hi0).Add(xi0.Mul(hr0)))
			im1 = im1.Add(xr1.Mul(hi1).Add(xi1.Mul(hr1)))
			im2 = im2.Add(xr2.Mul(hi2).Add(xi2.Mul(hr2)))
			im3 = im3.Add(xr3.Mul(hi3).Add(xi3.Mul(hr3)))
		}
		afirSt8(&s.Re, k, re0, re1)
		afirSt8(&s.Re, k+8, re2, re3)
		afirSt8(&s.Im, k, im0, im1)
		afirSt8(&s.Im, k+8, im2, im3)
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

// afirAdaptTerms is afirAdaptTermsGo with sixteen bins of g held in registers
// across the pairs.
func afirAdaptTerms(xs, hs []*FftData, g *FftData) {
	hs = hs[:len(xs)]
	for k := 0; k < FftLengthBy2; k += 16 {
		gr0, gr1, gr2, gr3 := afirLd16(&g.Re, k)
		gi0, gi1, gi2, gi3 := afirLd16(&g.Im, k)
		for t, x := range xs {
			h := hs[t]
			xr0, xr1, xr2, xr3 := afirLd16(&x.Re, k)
			xi0, xi1, xi2, xi3 := afirLd16(&x.Im, k)
			hr0, hr1, hr2, hr3 := afirLd16(&h.Re, k)
			hi0, hi1, hi2, hi3 := afirLd16(&h.Im, k)
			afirSt8(&h.Re, k, hr0.Add(xr0.Mul(gr0).Add(xi0.Mul(gi0))), hr1.Add(xr1.Mul(gr1).Add(xi1.Mul(gi1))))
			afirSt8(&h.Re, k+8, hr2.Add(xr2.Mul(gr2).Add(xi2.Mul(gi2))), hr3.Add(xr3.Mul(gr3).Add(xi3.Mul(gi3))))
			afirSt8(&h.Im, k, hi0.Add(xr0.Mul(gi0).Sub(xi0.Mul(gr0))), hi1.Add(xr1.Mul(gi1).Sub(xi1.Mul(gr1))))
			afirSt8(&h.Im, k+8, hi2.Add(xr2.Mul(gi2).Sub(xi2.Mul(gr2))), hi3.Add(xr3.Mul(gi3).Sub(xi3.Mul(gr3))))
		}
	}
	for t, x := range xs {
		afirAdaptBinsGo(x, g, hs[t], FftLengthBy2, FftLengthBy2Plus1)
	}
}

// afirLd16 loads bins k..k+15 as four vectors.
func afirLd16(a *[FftLengthBy2Plus1]float32, k int) (v0, v1, v2, v3 archsimd.Float32x4) {
	p := (*[16]float32)(a[k : k+16])
	return archsimd.LoadFloat32x4Array((*[4]float32)(p[:4])), archsimd.LoadFloat32x4Array((*[4]float32)(p[4:8])),
		archsimd.LoadFloat32x4Array((*[4]float32)(p[8:12])), archsimd.LoadFloat32x4Array((*[4]float32)(p[12:]))
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

// afirHasNeonFreqResp reports whether afirFreqRespPartitionNEON exists.
const afirHasNeonFreqResp = true

// afirFreqRespPartitionNEON computes one partition of
// ComputeFrequencyResponse_Neon: h2 = +0, then the maximum with the response
// of every channel in order, eight bins held in registers across the channels.
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

// afirMaxAbsUpdate is afirMaxAbsUpdateGo. The ordered compare is false for a
// NaN on either side, which keeps ir.
func afirMaxAbsUpdate(ir, h *[FftLengthBy2]float32) {
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[4]float32)(ir[k : k+4])
		a := archsimd.LoadFloat32x4Array(p)
		b := archsimd.LoadFloat32x4Array((*[4]float32)(h[k : k+4]))
		b.IfElse(a.Abs().Less(b.Abs()), a).StoreArray(p)
	}
}

// afirHasX86FreqResp reports whether afirFreqRespPartitionX86 runs the lane.
func afirHasX86FreqResp(Aec3Optimization) bool { return false }

func afirFreqRespPartitionX86(Aec3Optimization, []FftData, *[FftLengthBy2Plus1]float32) {
	panic("aec3: no x86 frequency response kernel")
}
