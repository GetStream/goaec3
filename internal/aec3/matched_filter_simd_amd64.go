//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// x86 kernels for MatchedFilterCore_SSE2 and MatchedFilterCore_AVX2. The
// 128-bit archsimd operations are VEX encoded (AVX); VMULPS/VADDPS on XMM
// registers round exactly like MULPS/ADDPS, so they reproduce the SSE2 lane.
// The NEON lane runs its pure-Go reference on amd64.
//
// The loops walk shrinking slices through fixed-size array views, so the
// compiler proves every load and store in bounds.

var (
	dlyUseAvx = archsimd.X86.AVX()
	dlyUseFma = archsimd.X86.AVX2() && archsimd.X86.FMA()
)

func dlyLd4(p *[4]float32) archsimd.Float32x4 { return archsimd.LoadFloat32x4Array(p) }

func dlyLd8(p *[8]float32) archsimd.Float32x8 { return archsimd.LoadFloat32x8Array(p) }

func dlyDotNeon(h, x []float32, acc *[8]float32) { dlyDotNeonRef(h, x, acc) }

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32, x2 *[4]float32) float32 {
	return dlyAccErrNeonRef(h, x, y, accErr, x2)
}

func dlyAxpyDotNeon(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	dlyAxpyDotNeonRef(h, xa, xb, alpha, acc)
}

func dlyDotSse2(h, x []float32, acc *[16]float32) {
	if !dlyUseAvx {
		dlyDotSse2Ref(h, x, acc)
		return
	}
	x = x[:len(h)]
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	x2 := dlyLd4((*[4]float32)(acc[8:12]))
	x24 := dlyLd4((*[4]float32)(acc[12:16]))
	for len(h) >= 8 && len(x) >= 8 {
		hh := (*[8]float32)(h)
		xx := (*[8]float32)(x)
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		hk := dlyLd4((*[4]float32)(hh[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		hk4 := dlyLd4((*[4]float32)(hh[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		s = s.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
		h = h[8:]
		x = x[8:]
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	s4.StoreArray((*[4]float32)(acc[4:8]))
	x2.StoreArray((*[4]float32)(acc[8:12]))
	x24.StoreArray((*[4]float32)(acc[12:16]))
}

// dlyAxpyDotSse2 equals dlyAxpyRef(h, xa, alpha) followed by
// dlyDotSse2Ref(h, xb, acc) for len(h) a multiple of 8.
func dlyAxpyDotSse2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	if !dlyUseAvx {
		dlyAxpyDotSse2Ref(h, xa, xb, alpha, acc)
		return
	}
	xa = xa[:len(h)]
	xb = xb[:len(h)]
	a := archsimd.BroadcastFloat32x4(alpha)
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	x2 := dlyLd4((*[4]float32)(acc[8:12]))
	x24 := dlyLd4((*[4]float32)(acc[12:16]))
	for len(h) >= 8 && len(xa) >= 8 && len(xb) >= 8 {
		hh := (*[8]float32)(h)
		aa := (*[8]float32)(xa)
		bb := (*[8]float32)(xb)
		hk := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(aa[0:4]))))
		hk4 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(aa[4:8]))))
		hk.StoreArray((*[4]float32)(hh[0:4]))
		hk4.StoreArray((*[4]float32)(hh[4:8]))
		xk := dlyLd4((*[4]float32)(bb[0:4]))
		xk4 := dlyLd4((*[4]float32)(bb[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		s = s.Add(hk.Mul(xk))
		s4 = s4.Add(hk4.Mul(xk4))
		h = h[8:]
		xa = xa[8:]
		xb = xb[8:]
	}
	s.StoreArray((*[4]float32)(acc[0:4]))
	s4.StoreArray((*[4]float32)(acc[4:8]))
	x2.StoreArray((*[4]float32)(acc[8:12]))
	x24.StoreArray((*[4]float32)(acc[12:16]))
}

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32, x2 *[8]float32) float32 {
	if !dlyUseAvx {
		return dlyAccErrSse2Ref(h, x, y, accErr, x2)
	}
	x = x[:len(h)]
	accErr = accErr[:len(h)/8*2]
	x2v := dlyLd4((*[4]float32)(x2[0:4]))
	x2v4 := dlyLd4((*[4]float32)(x2[4:8]))
	var s float32
	for len(h) >= 8 && len(x) >= 8 && len(accErr) >= 2 {
		hh := (*[8]float32)(h)
		xx := (*[8]float32)(x)
		ae := (*[2]float32)(accErr)
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		hk := dlyLd4((*[4]float32)(hh[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		hk4 := dlyLd4((*[4]float32)(hh[4:8]))
		x2v = x2v.Add(xk.Mul(xk))
		x2v4 = x2v4.Add(xk4.Mul(xk4))
		var p, q [4]float32
		hk.Mul(xk).StoreArray(&p)
		hk4.Mul(xk4).StoreArray(&q)
		s += ((p[0] + p[1]) + p[2]) + p[3]
		e0 := s - y
		s += ((q[0] + q[1]) + q[2]) + q[3]
		e1 := s - y
		ae[0] += float32(e0 * e0)
		ae[1] += float32(e1 * e1)
		h = h[8:]
		x = x[8:]
		accErr = accErr[2:]
	}
	x2v.StoreArray((*[4]float32)(x2[0:4]))
	x2v4.StoreArray((*[4]float32)(x2[4:8]))
	return s
}

func dlyAxpy(h, x []float32, alpha float32) {
	if !dlyUseAvx {
		dlyAxpyRef(h, x, alpha)
		return
	}
	x = x[:len(h)]
	a := archsimd.BroadcastFloat32x4(alpha)
	for len(h) >= 8 && len(x) >= 8 {
		hh := (*[8]float32)(h)
		xx := (*[8]float32)(x)
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(xx[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
		h = h[8:]
		x = x[8:]
	}
	for len(h) >= 4 && len(x) >= 4 {
		hp := (*[4]float32)(h)
		dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(x)))).StoreArray(hp)
		h = h[4:]
		x = x[4:]
	}
	x = x[:len(h)]
	for k := range h {
		h[k] += float32(alpha * x[k])
	}
}

func dlyDotAvx2(h, x []float32, acc *[32]float32) {
	if !dlyUseFma {
		dlyDotAvx2Ref(h, x, acc)
		return
	}
	x = x[:len(h)]
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	x2 := dlyLd8((*[8]float32)(acc[16:24]))
	x28 := dlyLd8((*[8]float32)(acc[24:32]))
	for len(h) >= 16 && len(x) >= 16 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		hk := dlyLd8((*[8]float32)(hh[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		hk8 := dlyLd8((*[8]float32)(hh[8:16]))
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
		s = hk.MulAdd(xk, s)
		s8 = hk8.MulAdd(xk8, s8)
		h = h[16:]
		x = x[16:]
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	x2.StoreArray((*[8]float32)(acc[16:24]))
	x28.StoreArray((*[8]float32)(acc[24:32]))
}

// dlyAxpyFmaDotAvx2 equals dlyAxpyFmaRef(h, xa, alpha) followed by
// dlyDotAvx2Ref(h, xb, acc) for len(h) a multiple of 16.
func dlyAxpyFmaDotAvx2(h, xa, xb []float32, alpha float32, acc *[32]float32) {
	if !dlyUseFma {
		dlyAxpyFmaDotAvx2Ref(h, xa, xb, alpha, acc)
		return
	}
	xa = xa[:len(h)]
	xb = xb[:len(h)]
	a := archsimd.BroadcastFloat32x8(alpha)
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	x2 := dlyLd8((*[8]float32)(acc[16:24]))
	x28 := dlyLd8((*[8]float32)(acc[24:32]))
	for len(h) >= 16 && len(xa) >= 16 && len(xb) >= 16 {
		hh := (*[16]float32)(h)
		aa := (*[16]float32)(xa)
		bb := (*[16]float32)(xb)
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
		s = hk.MulAdd(xk, s)
		s8 = hk8.MulAdd(xk8, s8)
		h = h[16:]
		xa = xa[16:]
		xb = xb[16:]
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	x2.StoreArray((*[8]float32)(acc[16:24]))
	x28.StoreArray((*[8]float32)(acc[24:32]))
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	if !dlyUseFma {
		return dlyAccErrAvx2Ref(h, x, y, accErr, x2)
	}
	x = x[:len(h)]
	accErr = accErr[:len(h)/16*4]
	x2v := dlyLd8((*[8]float32)(x2[0:8]))
	x2v8 := dlyLd8((*[8]float32)(x2[8:16]))
	var s float32
	for len(h) >= 16 && len(x) >= 16 && len(accErr) >= 4 {
		hh := (*[16]float32)(h)
		xx := (*[16]float32)(x)
		ap := (*[4]float32)(accErr)
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		hk := dlyLd8((*[8]float32)(hh[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		hk8 := dlyLd8((*[8]float32)(hh[8:16]))
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
		ev := archsimd.LoadFloat32x4Array(&e)
		ev.MulAdd(ev, dlyLd4(ap)).StoreArray(ap)
		h = h[16:]
		x = x[16:]
		accErr = accErr[4:]
	}
	x2v.StoreArray((*[8]float32)(x2[0:8]))
	x2v8.StoreArray((*[8]float32)(x2[8:16]))
	return s
}

func dlyAxpyFma(h, x []float32, alpha float32) {
	if !dlyUseFma {
		dlyAxpyFmaRef(h, x, alpha)
		return
	}
	x = x[:len(h)]
	a := archsimd.BroadcastFloat32x8(alpha)
	for len(h) >= 8 && len(x) >= 8 {
		hp := (*[8]float32)(h)
		dlyLd8((*[8]float32)(x)).MulAdd(a, dlyLd8(hp)).StoreArray(hp)
		h = h[8:]
		x = x[8:]
	}
	x = x[:len(h)]
	for k := range h {
		h[k] = dlyFmaf(x[k], alpha, h[k])
	}
}

var dlyPeakSimd = dlyUseAvx
