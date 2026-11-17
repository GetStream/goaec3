//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// NEON kernels for MatchedFilterCore_NEON. Mul followed by Add compiles to
// separate FMUL/FADD, matching the unfused vmlaq_f32 of the C++. The SSE2 and
// AVX2 lanes run their pure-Go references on arm64.

func dlyDotNeon(h, x []float32, acc *[8]float32) {
	n := len(h) &^ 3
	x = x[:n]
	s := archsimd.LoadFloat32x4(acc[0:4])
	x2 := archsimd.LoadFloat32x4(acc[4:8])
	for k := 0; k < n; k += 4 {
		xk := archsimd.LoadFloat32x4(x[k : k+4])
		hk := archsimd.LoadFloat32x4(h[k : k+4])
		x2 = x2.Add(xk.Mul(xk))
		s = s.Add(hk.Mul(xk))
	}
	s.Store(acc[0:4])
	x2.Store(acc[4:8])
}

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32, x2 *[4]float32) float32 {
	n := len(h) &^ 3
	x = x[:n]
	accErr = accErr[:n/4]
	x2v := archsimd.LoadFloat32x4(x2[:])
	var s float32
	for g := range accErr {
		k := 4 * g
		xk := archsimd.LoadFloat32x4(x[k : k+4])
		hk := archsimd.LoadFloat32x4(h[k : k+4])
		x2v = x2v.Add(xk.Mul(xk))
		p := hk.Mul(xk)
		// SumAllElements: vpadd(lo, hi) then vpadd(sum, sum).
		r := p.ConcatAddPairs(p)
		r = r.ConcatAddPairs(r)
		s += r.GetElem(0)
		e := s - y
		accErr[g] += float32(e * e)
	}
	x2v.Store(x2[:])
	return s
}

func dlyAxpy(h, x []float32, alpha float32) {
	n := len(h) &^ 3
	x = x[:len(h)]
	a := archsimd.BroadcastFloat32x4(alpha)
	for k := 0; k < n; k += 4 {
		hk := archsimd.LoadFloat32x4(h[k : k+4])
		xk := archsimd.LoadFloat32x4(x[k : k+4])
		hk.Add(a.Mul(xk)).Store(h[k : k+4])
	}
	for k := n; k < len(h); k++ {
		h[k] += float32(alpha * x[k])
	}
}

func dlyDotSse2(h, x []float32, acc *[16]float32) { dlyDotSse2Ref(h, x, acc) }

func dlyDotAvx2(h, x []float32, acc *[32]float32) { dlyDotAvx2Ref(h, x, acc) }

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32, x2 *[8]float32) float32 {
	return dlyAccErrSse2Ref(h, x, y, accErr, x2)
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	return dlyAccErrAvx2Ref(h, x, y, accErr, x2)
}

func dlyAxpyFma(h, x []float32, alpha float32) {
	n := len(h) &^ 3
	x = x[:len(h)]
	a := archsimd.BroadcastFloat32x4(alpha)
	for k := 0; k < n; k += 4 {
		hk := archsimd.LoadFloat32x4(h[k : k+4])
		xk := archsimd.LoadFloat32x4(x[k : k+4])
		xk.MulAdd(a, hk).Store(h[k : k+4])
	}
	for k := n; k < len(h); k++ {
		h[k] = dlyFmaf(x[k], alpha, h[k])
	}
}
