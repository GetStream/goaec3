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
		a := archsimd.LoadFloat32x4Array(p)
		a.IfElse(a.Greater(n), n).StoreArray(p) // _mm_max_ps(a, n) = a > n ? a : n
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
	archsimd.ClearAVXUpperBits()
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
	archsimd.ClearAVXUpperBits()
}

// afirFreqRespBlockAVX2: re2 = re*re; re2 = _mm256_fmadd_ps(im, im, re2).
func afirFreqRespBlockAVX2(h *FftData, h2 *[FftLengthBy2Plus1]float32) {
	for j := 0; j < FftLengthBy2; j += 8 {
		re := archsimd.LoadFloat32x8Array((*[8]float32)(h.Re[j : j+8]))
		im := archsimd.LoadFloat32x8Array((*[8]float32)(h.Im[j : j+8]))
		p := (*[8]float32)(h2[j : j+8])
		n := im.MulAdd(im, re.Mul(re))
		a := archsimd.LoadFloat32x8Array(p)
		a.IfElse(a.Greater(n), n).StoreArray(p) // _mm256_max_ps(a, n) = a > n ? a : n
	}
	archsimd.ClearAVXUpperBits()
}

func afirErlBlockAVX2(h2, erl *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k += 8 {
		p := (*[8]float32)(erl[k : k+8])
		archsimd.LoadFloat32x8Array(p).Add(archsimd.LoadFloat32x8Array((*[8]float32)(h2[k : k+8]))).StoreArray(p)
	}
	archsimd.ClearAVXUpperBits()
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
	case opt == OptSse2:
		afirFreqRespBlockSse2Go(h, h2)
	case opt == OptNeon:
		afirFreqRespBlockNeonGo(h, h2)
	default:
		afirFreqRespBinsGo(h, h2, 0, FftLengthBy2)
	}
}

// afirHasTermKernels reports that afirApplyTerms/afirAdaptTerms are archsimd
// kernels. Every lane computes the same rounded products and per-bin
// accumulation order, so one kernel serves the NEON, SSE2 and AVX2 lanes;
// 256-bit code needs AVX2, 128-bit code AVX.
var afirHasTermKernels = afirHasAVX

// afirApplyTerms is afirApplyTermsGo with the bin loop outside the pair loop:
// a block of S stays in registers while all pairs accumulate into it, in
// pair order, starting from +0.
func afirApplyTerms(xs, hs []*FftData, s *FftData) {
	hs = hs[:len(xs)]
	if afirHasAVX2 {
		var zero archsimd.Float32x8
		block := func(k int) {
			_ = zero // a captured vector makes the compiler inline the closure
			var re0, re1, im0, im1 archsimd.Float32x8
			for t, x := range xs {
				h := hs[t]
				xr0 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Re[k : k+8]))
				xr1 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Re[k+8 : k+16]))
				xi0 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Im[k : k+8]))
				xi1 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Im[k+8 : k+16]))
				hr0 := archsimd.LoadFloat32x8Array((*[8]float32)(h.Re[k : k+8]))
				hr1 := archsimd.LoadFloat32x8Array((*[8]float32)(h.Re[k+8 : k+16]))
				hi0 := archsimd.LoadFloat32x8Array((*[8]float32)(h.Im[k : k+8]))
				hi1 := archsimd.LoadFloat32x8Array((*[8]float32)(h.Im[k+8 : k+16]))
				re0 = re0.Add(xr0.Mul(hr0).Sub(xi0.Mul(hi0)))
				re1 = re1.Add(xr1.Mul(hr1).Sub(xi1.Mul(hi1)))
				im0 = im0.Add(xr0.Mul(hi0).Add(xi0.Mul(hr0)))
				im1 = im1.Add(xr1.Mul(hi1).Add(xi1.Mul(hr1)))
			}
			re0.StoreArray((*[8]float32)(s.Re[k : k+8]))
			re1.StoreArray((*[8]float32)(s.Re[k+8 : k+16]))
			im0.StoreArray((*[8]float32)(s.Im[k : k+8]))
			im1.StoreArray((*[8]float32)(s.Im[k+8 : k+16]))
		}
		block(0)
		block(16)
		block(32)
		block(48)
		archsimd.ClearAVXUpperBits()
	} else {
		var zero archsimd.Float32x4
		block := func(k int) {
			_ = zero // a captured vector makes the compiler inline the closure
			var re0, re1, im0, im1 archsimd.Float32x4
			for t, x := range xs {
				h := hs[t]
				xr0 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k : k+4]))
				xr1 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k+4 : k+8]))
				xi0 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k : k+4]))
				xi1 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k+4 : k+8]))
				hr0 := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[k : k+4]))
				hr1 := archsimd.LoadFloat32x4Array((*[4]float32)(h.Re[k+4 : k+8]))
				hi0 := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[k : k+4]))
				hi1 := archsimd.LoadFloat32x4Array((*[4]float32)(h.Im[k+4 : k+8]))
				re0 = re0.Add(xr0.Mul(hr0).Sub(xi0.Mul(hi0)))
				re1 = re1.Add(xr1.Mul(hr1).Sub(xi1.Mul(hi1)))
				im0 = im0.Add(xr0.Mul(hi0).Add(xi0.Mul(hr0)))
				im1 = im1.Add(xr1.Mul(hi1).Add(xi1.Mul(hr1)))
			}
			re0.StoreArray((*[4]float32)(s.Re[k : k+4]))
			re1.StoreArray((*[4]float32)(s.Re[k+4 : k+8]))
			im0.StoreArray((*[4]float32)(s.Im[k : k+4]))
			im1.StoreArray((*[4]float32)(s.Im[k+4 : k+8]))
		}
		block(0)
		block(8)
		block(16)
		block(24)
		block(32)
		block(40)
		block(48)
		block(56)
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

// afirAdaptTerms is afirAdaptTermsGo with a block of G held in registers
// across all pairs.
func afirAdaptTerms(xs, hs []*FftData, g *FftData) {
	hs = hs[:len(xs)]
	if afirHasAVX2 {
		var zero archsimd.Float32x8
		block := func(k int) {
			_ = zero // a captured vector makes the compiler inline the closure
			gr0 := archsimd.LoadFloat32x8Array((*[8]float32)(g.Re[k : k+8]))
			gr1 := archsimd.LoadFloat32x8Array((*[8]float32)(g.Re[k+8 : k+16]))
			gi0 := archsimd.LoadFloat32x8Array((*[8]float32)(g.Im[k : k+8]))
			gi1 := archsimd.LoadFloat32x8Array((*[8]float32)(g.Im[k+8 : k+16]))
			for t, x := range xs {
				h := hs[t]
				xr0 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Re[k : k+8]))
				xr1 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Re[k+8 : k+16]))
				xi0 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Im[k : k+8]))
				xi1 := archsimd.LoadFloat32x8Array((*[8]float32)(x.Im[k+8 : k+16]))
				hr0 := (*[8]float32)(h.Re[k : k+8])
				hr1 := (*[8]float32)(h.Re[k+8 : k+16])
				hi0 := (*[8]float32)(h.Im[k : k+8])
				hi1 := (*[8]float32)(h.Im[k+8 : k+16])
				archsimd.LoadFloat32x8Array(hr0).Add(xr0.Mul(gr0).Add(xi0.Mul(gi0))).StoreArray(hr0)
				archsimd.LoadFloat32x8Array(hr1).Add(xr1.Mul(gr1).Add(xi1.Mul(gi1))).StoreArray(hr1)
				archsimd.LoadFloat32x8Array(hi0).Add(xr0.Mul(gi0).Sub(xi0.Mul(gr0))).StoreArray(hi0)
				archsimd.LoadFloat32x8Array(hi1).Add(xr1.Mul(gi1).Sub(xi1.Mul(gr1))).StoreArray(hi1)
			}
		}
		block(0)
		block(16)
		block(32)
		block(48)
		archsimd.ClearAVXUpperBits()
	} else {
		var zero archsimd.Float32x4
		block := func(k int) {
			_ = zero // a captured vector makes the compiler inline the closure
			gr0 := archsimd.LoadFloat32x4Array((*[4]float32)(g.Re[k : k+4]))
			gr1 := archsimd.LoadFloat32x4Array((*[4]float32)(g.Re[k+4 : k+8]))
			gi0 := archsimd.LoadFloat32x4Array((*[4]float32)(g.Im[k : k+4]))
			gi1 := archsimd.LoadFloat32x4Array((*[4]float32)(g.Im[k+4 : k+8]))
			for t, x := range xs {
				h := hs[t]
				xr0 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k : k+4]))
				xr1 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Re[k+4 : k+8]))
				xi0 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k : k+4]))
				xi1 := archsimd.LoadFloat32x4Array((*[4]float32)(x.Im[k+4 : k+8]))
				hr0 := (*[4]float32)(h.Re[k : k+4])
				hr1 := (*[4]float32)(h.Re[k+4 : k+8])
				hi0 := (*[4]float32)(h.Im[k : k+4])
				hi1 := (*[4]float32)(h.Im[k+4 : k+8])
				archsimd.LoadFloat32x4Array(hr0).Add(xr0.Mul(gr0).Add(xi0.Mul(gi0))).StoreArray(hr0)
				archsimd.LoadFloat32x4Array(hr1).Add(xr1.Mul(gr1).Add(xi1.Mul(gi1))).StoreArray(hr1)
				archsimd.LoadFloat32x4Array(hi0).Add(xr0.Mul(gi0).Sub(xi0.Mul(gr0))).StoreArray(hi0)
				archsimd.LoadFloat32x4Array(hi1).Add(xr1.Mul(gi1).Sub(xi1.Mul(gr1))).StoreArray(hi1)
			}
		}
		block(0)
		block(8)
		block(16)
		block(24)
		block(32)
		block(40)
		block(48)
		block(56)
	}
	for t, x := range xs {
		afirAdaptBinsGo(x, g, hs[t], FftLengthBy2, FftLengthBy2Plus1)
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

// afirHasNeonFreqResp is false: the NEON lane runs afirFreqRespBlock per
// channel.
const afirHasNeonFreqResp = false

func afirFreqRespPartitionNEON(hs []FftData, h2 *[FftLengthBy2Plus1]float32) {
	panic("aec3: no NEON frequency response kernel")
}

// afirMaxAbsUpdate is afirMaxAbsUpdateGo: the ordered compare of the absolute
// values is false for a NaN on either side, which keeps ir.
func afirMaxAbsUpdate(ir, h *[FftLengthBy2]float32) {
	if !afirHasAVX {
		afirMaxAbsUpdateGo(ir, h)
		return
	}
	// Float32x4.Abs builds its mask with VPBROADCASTD, an AVX2 instruction.
	abs := kernSplatI4(0x7fffffff)
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[4]float32)(ir[k : k+4])
		a := archsimd.LoadFloat32x4Array(p)
		b := archsimd.LoadFloat32x4Array((*[4]float32)(h[k : k+4]))
		aa := a.AsInt32x4().And(abs).AsFloat32x4()
		ba := b.AsInt32x4().And(abs).AsFloat32x4()
		b.IfElse(aa.Less(ba), a).StoreArray(p)
	}
}

// afirHasX86FreqResp reports whether afirFreqRespPartitionX86 runs the
// lane: SSE2 with AVX (VEX encoding), AVX2 with FMA.
func afirHasX86FreqResp(opt Aec3Optimization) bool {
	return opt == OptSse2 && afirHasAVX || opt == OptAvx2 && afirHasAVX2FMA
}

// afirFreqRespPartitionX86 computes one partition of
// ComputeFrequencyResponse_Sse2 or _Avx2: H2 = +0, then
// H2 = _mm_max_ps(H2, |H[ch]|^2) for every channel in order (bin 64 with
// std::max), with sixteen bins held in registers across the channels
// instead of being cleared, loaded and stored per channel. It tests the CPU
// features itself, as afirHasX86FreqResp does, so that every vector
// instruction follows a test of its feature (TestSimdInstructionsGated).
func afirFreqRespPartitionX86(opt Aec3Optimization, hs []FftData, h2 *[FftLengthBy2Plus1]float32) {
	switch {
	case opt == OptAvx2 && afirHasAVX2FMA:
		var zero archsimd.Float32x8
		block := func(k int) {
			_ = zero // a captured vector makes the compiler inline the closure
			var a0, a1 archsimd.Float32x8
			for c := range hs {
				h := &hs[c]
				re := (*[16]float32)(h.Re[k : k+16])
				im := (*[16]float32)(h.Im[k : k+16])
				re0, re1 := archsimd.LoadFloat32x8Array((*[8]float32)(re[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(re[8:16]))
				im0, im1 := archsimd.LoadFloat32x8Array((*[8]float32)(im[0:8])), archsimd.LoadFloat32x8Array((*[8]float32)(im[8:16]))
				n0 := im0.MulAdd(im0, re0.Mul(re0))
				n1 := im1.MulAdd(im1, re1.Mul(re1))
				a0 = a0.IfElse(a0.Greater(n0), n0)
				a1 = a1.IfElse(a1.Greater(n1), n1)
			}
			o := (*[16]float32)(h2[k : k+16])
			a0.StoreArray((*[8]float32)(o[0:8]))
			a1.StoreArray((*[8]float32)(o[8:16]))
		}
		block(0)
		block(16)
		block(32)
		block(48)
		archsimd.ClearAVXUpperBits()
	case opt == OptSse2 && afirHasAVX:
		var zero archsimd.Float32x4
		block := func(k int) {
			_ = zero // a captured vector makes the compiler inline the closure
			var a0, a1 archsimd.Float32x4
			for c := range hs {
				h := &hs[c]
				re := (*[8]float32)(h.Re[k : k+8])
				im := (*[8]float32)(h.Im[k : k+8])
				re0, re1 := archsimd.LoadFloat32x4Array((*[4]float32)(re[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(re[4:8]))
				im0, im1 := archsimd.LoadFloat32x4Array((*[4]float32)(im[0:4])), archsimd.LoadFloat32x4Array((*[4]float32)(im[4:8]))
				n0 := re0.Mul(re0).Add(im0.Mul(im0))
				n1 := re1.Mul(re1).Add(im1.Mul(im1))
				a0 = a0.IfElse(a0.Greater(n0), n0)
				a1 = a1.IfElse(a1.Greater(n1), n1)
			}
			o := (*[8]float32)(h2[k : k+8])
			a0.StoreArray((*[4]float32)(o[0:4]))
			a1.StoreArray((*[4]float32)(o[4:8]))
		}
		block(0)
		block(8)
		block(16)
		block(24)
		block(32)
		block(40)
		block(48)
		block(56)
	default:
		panic("aec3: afirFreqRespPartitionX86 without afirHasX86FreqResp")
	}
	h2[FftLengthBy2] = 0
	for c := range hs {
		afirFreqRespBinsGo(&hs[c], h2, FftLengthBy2, FftLengthBy2Plus1)
	}
}
