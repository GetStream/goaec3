//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// x86 kernels for MatchedFilterCore_SSE2 and MatchedFilterCore_AVX2. The
// 128-bit archsimd operations are VEX encoded (AVX); VMULPS/VADDPS on XMM
// registers round exactly like MULPS/ADDPS, so they reproduce the SSE2 lane.
// The NEON lane runs its pure-Go reference on amd64.

var (
	dlyUseAvx = archsimd.X86.AVX()
	dlyUseFma = archsimd.X86.AVX2() && archsimd.X86.FMA()
)

func dlyDotNeon(h, x []float32, acc *[8]float32) { dlyDotNeonRef(h, x, acc) }

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32, x2 *[4]float32) float32 {
	return dlyAccErrNeonRef(h, x, y, accErr, x2)
}

func dlyDotSse2(h, x []float32, acc *[16]float32) {
	if !dlyUseAvx {
		dlyDotSse2Ref(h, x, acc)
		return
	}
	n := len(h) &^ 7
	x = x[:n]
	s := archsimd.LoadFloat32x4(acc[0:4])
	s4 := archsimd.LoadFloat32x4(acc[4:8])
	x2 := archsimd.LoadFloat32x4(acc[8:12])
	x24 := archsimd.LoadFloat32x4(acc[12:16])
	for k := 0; k < n; k += 8 {
		xk := archsimd.LoadFloat32x4(x[k : k+4])
		hk := archsimd.LoadFloat32x4(h[k : k+4])
		xk4 := archsimd.LoadFloat32x4(x[k+4 : k+8])
		hk4 := archsimd.LoadFloat32x4(h[k+4 : k+8])
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		s = s.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
	}
	s.Store(acc[0:4])
	s4.Store(acc[4:8])
	x2.Store(acc[8:12])
	x24.Store(acc[12:16])
}

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32, x2 *[8]float32) float32 {
	if !dlyUseAvx {
		return dlyAccErrSse2Ref(h, x, y, accErr, x2)
	}
	n := len(h) &^ 7
	x = x[:n]
	accErr = accErr[:n/4]
	x2v := archsimd.LoadFloat32x4(x2[0:4])
	x2v4 := archsimd.LoadFloat32x4(x2[4:8])
	var s float32
	for g := 0; g < n/8; g++ {
		k := 8 * g
		xk := archsimd.LoadFloat32x4(x[k : k+4])
		hk := archsimd.LoadFloat32x4(h[k : k+4])
		xk4 := archsimd.LoadFloat32x4(x[k+4 : k+8])
		hk4 := archsimd.LoadFloat32x4(h[k+4 : k+8])
		x2v = x2v.Add(xk.Mul(xk))
		x2v4 = x2v4.Add(xk4.Mul(xk4))
		p := hk.Mul(xk)
		q := hk4.Mul(xk4)
		s += ((p.GetElem(0) + p.GetElem(1)) + p.GetElem(2)) + p.GetElem(3)
		e0 := s - y
		s += ((q.GetElem(0) + q.GetElem(1)) + q.GetElem(2)) + q.GetElem(3)
		e1 := s - y
		accErr[2*g] += float32(e0 * e0)
		accErr[2*g+1] += float32(e1 * e1)
	}
	x2v.Store(x2[0:4])
	x2v4.Store(x2[4:8])
	return s
}

func dlyAxpy(h, x []float32, alpha float32) {
	if !dlyUseAvx {
		dlyAxpyRef(h, x, alpha)
		return
	}
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

func dlyDotAvx2(h, x []float32, acc *[32]float32) {
	if !dlyUseFma {
		dlyDotAvx2Ref(h, x, acc)
		return
	}
	n := len(h) &^ 15
	x = x[:n]
	s := archsimd.LoadFloat32x8(acc[0:8])
	s8 := archsimd.LoadFloat32x8(acc[8:16])
	x2 := archsimd.LoadFloat32x8(acc[16:24])
	x28 := archsimd.LoadFloat32x8(acc[24:32])
	for k := 0; k < n; k += 16 {
		xk := archsimd.LoadFloat32x8(x[k : k+8])
		hk := archsimd.LoadFloat32x8(h[k : k+8])
		xk8 := archsimd.LoadFloat32x8(x[k+8 : k+16])
		hk8 := archsimd.LoadFloat32x8(h[k+8 : k+16])
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
		s = hk.MulAdd(xk, s)
		s8 = hk8.MulAdd(xk8, s8)
	}
	s.Store(acc[0:8])
	s8.Store(acc[8:16])
	x2.Store(acc[16:24])
	x28.Store(acc[24:32])
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	if !dlyUseFma {
		return dlyAccErrAvx2Ref(h, x, y, accErr, x2)
	}
	n := len(h) &^ 15
	x = x[:n]
	accErr = accErr[:n/4]
	x2v := archsimd.LoadFloat32x8(x2[0:8])
	x2v8 := archsimd.LoadFloat32x8(x2[8:16])
	var s float32
	for g := 0; g < n/16; g++ {
		k := 16 * g
		xk := archsimd.LoadFloat32x8(x[k : k+8])
		hk := archsimd.LoadFloat32x8(h[k : k+8])
		xk8 := archsimd.LoadFloat32x8(x[k+8 : k+16])
		hk8 := archsimd.LoadFloat32x8(h[k+8 : k+16])
		x2v = xk.MulAdd(xk, x2v)
		x2v8 = xk8.MulAdd(xk8, x2v8)
		p := hk.Mul(xk)
		q := hk8.Mul(xk8)
		r := p.ConcatAddPairsGrouped(q)
		r = r.ConcatAddPairsGrouped(r)
		lo := r.GetLo()
		hi := r.GetHi()
		var e [4]float32
		s += lo.GetElem(0)
		e[0] = s - y
		s += hi.GetElem(0)
		e[1] = s - y
		s += lo.GetElem(1)
		e[2] = s - y
		s += hi.GetElem(1)
		e[3] = s - y
		ev := archsimd.LoadFloat32x4(e[:])
		a := accErr[4*g : 4*g+4]
		ev.MulAdd(ev, archsimd.LoadFloat32x4(a)).Store(a)
	}
	x2v.Store(x2[0:8])
	x2v8.Store(x2[8:16])
	return s
}

func dlyAxpyFma(h, x []float32, alpha float32) {
	if !dlyUseFma {
		dlyAxpyFmaRef(h, x, alpha)
		return
	}
	n := len(h) &^ 7
	x = x[:len(h)]
	a := archsimd.BroadcastFloat32x8(alpha)
	for k := 0; k < n; k += 8 {
		hk := archsimd.LoadFloat32x8(h[k : k+8])
		xk := archsimd.LoadFloat32x8(x[k : k+8])
		xk.MulAdd(a, hk).Store(h[k : k+8])
	}
	for k := n; k < len(h); k++ {
		h[k] = dlyFmaf(x[k], alpha, h[k])
	}
}
