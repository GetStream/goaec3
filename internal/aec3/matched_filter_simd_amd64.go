//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// x86 kernels for MatchedFilterCore_SSE2 and MatchedFilterCore_AVX2. The
// 128-bit archsimd operations are VEX encoded (AVX); VMULPS/VADDPS on XMM
// registers round exactly like MULPS/ADDPS, so they reproduce the SSE2 lane.
// The NEON lane runs its pure-Go reference on amd64.
//
// The loops index fixed-size array views of slices resliced to a common
// length and capacity (x[:n:n]) with an index bounded by len-15 (or len-7),
// so the compiler proves every access in bounds and the loop carries a
// single induction variable. fma(a, b, c) is symmetric in a and b, and
// VFMADD213PS overwrites the receiver, so the receiver of each MulAdd is the
// multiplicand that dies there.

var (
	dlyUseAvx = archsimd.X86.AVX()
	dlyUseFma = archsimd.X86.AVX2() && archsimd.X86.FMA()
)

// dlyCoreLane runs the MatchedFilterCore loop of the SIMD lane with group
// size g. On amd64 the update of a sample is fused into the next sample's
// dot product only when both cover h in one stretch (see dlyCoreSimple).
func dlyCoreLane(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	dlyCoreSimple(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
}

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
	n := len(h)
	x = x[:n:n]
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	x2 := dlyLd4((*[4]float32)(acc[8:12]))
	x24 := dlyLd4((*[4]float32)(acc[12:16]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		x2 = x2.Add(xk.Mul(xk))
		x24 = x24.Add(xk4.Mul(xk4))
		s = s.Add(dlyLd4((*[4]float32)(hh[0:4])).Mul(xk))
		s4 = s4.Add(dlyLd4((*[4]float32)(hh[4:8])).Mul(xk4))
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
	n := len(h)
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	s := dlyLd4((*[4]float32)(acc[0:4]))
	s4 := dlyLd4((*[4]float32)(acc[4:8]))
	x2 := dlyLd4((*[4]float32)(acc[8:12]))
	x24 := dlyLd4((*[4]float32)(acc[12:16]))
	for i := 0; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		aa := (*[8]float32)(xa[i : i+8])
		bb := (*[8]float32)(xb[i : i+8])
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
	n := len(h) &^ 7
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: n/4 : n/4]
	x2v := dlyLd4((*[4]float32)(x2[0:4]))
	x2v4 := dlyLd4((*[4]float32)(x2[4:8]))
	var s float32
	for g := 0; g < len(accErr)-1; g += 2 {
		i := 4 * g
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		ae := (*[2]float32)(accErr[g : g+2])
		xk := dlyLd4((*[4]float32)(xx[0:4]))
		xk4 := dlyLd4((*[4]float32)(xx[4:8]))
		x2v = x2v.Add(xk.Mul(xk))
		x2v4 = x2v4.Add(xk4.Mul(xk4))
		var p, q [4]float32
		dlyLd4((*[4]float32)(hh[0:4])).Mul(xk).StoreArray(&p)
		dlyLd4((*[4]float32)(hh[4:8])).Mul(xk4).StoreArray(&q)
		s += ((p[0] + p[1]) + p[2]) + p[3]
		e0 := s - y
		s += ((q[0] + q[1]) + q[2]) + q[3]
		e1 := s - y
		ae[0] += float32(e0 * e0)
		ae[1] += float32(e1 * e1)
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
	n := len(h)
	x = x[:n:n]
	a := archsimd.BroadcastFloat32x4(alpha)
	i := 0
	for ; i < n-7; i += 8 {
		hh := (*[8]float32)(h[i : i+8])
		xx := (*[8]float32)(x[i : i+8])
		h0 := dlyLd4((*[4]float32)(hh[0:4])).Add(a.Mul(dlyLd4((*[4]float32)(xx[0:4]))))
		h1 := dlyLd4((*[4]float32)(hh[4:8])).Add(a.Mul(dlyLd4((*[4]float32)(xx[4:8]))))
		h0.StoreArray((*[4]float32)(hh[0:4]))
		h1.StoreArray((*[4]float32)(hh[4:8]))
	}
	for ; i < n-3; i += 4 {
		hp := (*[4]float32)(h[i : i+4])
		dlyLd4(hp).Add(a.Mul(dlyLd4((*[4]float32)(x[i : i+4])))).StoreArray(hp)
	}
	for ; i < n; i++ {
		h[i] += float32(alpha * x[i])
	}
}

func dlyDotAvx2(h, x []float32, acc *[32]float32) {
	if !dlyUseFma {
		dlyDotAvx2Ref(h, x, acc)
		return
	}
	n := len(h)
	x = x[:n:n]
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	x2 := dlyLd8((*[8]float32)(acc[16:24]))
	x28 := dlyLd8((*[8]float32)(acc[24:32]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		xx := (*[16]float32)(x[i : i+16])
		xk := dlyLd8((*[8]float32)(xx[0:8]))
		xk8 := dlyLd8((*[8]float32)(xx[8:16]))
		s = dlyLd8((*[8]float32)(hh[0:8])).MulAdd(xk, s)
		s8 = dlyLd8((*[8]float32)(hh[8:16])).MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	x2.StoreArray((*[8]float32)(acc[16:24]))
	x28.StoreArray((*[8]float32)(acc[24:32]))
	archsimd.ClearAVXUpperBits()
}

// dlyAxpyFmaDotAvx2 equals dlyAxpyFmaRef(h, xa, alpha) followed by
// dlyDotAvx2Ref(h, xb, acc) for len(h) a multiple of 16.
func dlyAxpyFmaDotAvx2(h, xa, xb []float32, alpha float32, acc *[32]float32) {
	if !dlyUseFma {
		dlyAxpyFmaDotAvx2Ref(h, xa, xb, alpha, acc)
		return
	}
	n := len(h)
	xa = xa[:n:n]
	xb = xb[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	s := dlyLd8((*[8]float32)(acc[0:8]))
	s8 := dlyLd8((*[8]float32)(acc[8:16]))
	x2 := dlyLd8((*[8]float32)(acc[16:24]))
	x28 := dlyLd8((*[8]float32)(acc[24:32]))
	for i := 0; i < n-15; i += 16 {
		hh := (*[16]float32)(h[i : i+16])
		aa := (*[16]float32)(xa[i : i+16])
		bb := (*[16]float32)(xb[i : i+16])
		hk := dlyLd8((*[8]float32)(aa[0:8])).MulAdd(a, dlyLd8((*[8]float32)(hh[0:8])))
		hk8 := dlyLd8((*[8]float32)(aa[8:16])).MulAdd(a, dlyLd8((*[8]float32)(hh[8:16])))
		hk.StoreArray((*[8]float32)(hh[0:8]))
		hk8.StoreArray((*[8]float32)(hh[8:16]))
		xk := dlyLd8((*[8]float32)(bb[0:8]))
		xk8 := dlyLd8((*[8]float32)(bb[8:16]))
		s = hk.MulAdd(xk, s)
		s8 = hk8.MulAdd(xk8, s8)
		x2 = xk.MulAdd(xk, x2)
		x28 = xk8.MulAdd(xk8, x28)
	}
	s.StoreArray((*[8]float32)(acc[0:8]))
	s8.StoreArray((*[8]float32)(acc[8:16]))
	x2.StoreArray((*[8]float32)(acc[16:24]))
	x28.StoreArray((*[8]float32)(acc[24:32]))
	archsimd.ClearAVXUpperBits()
}

// dlyAccErrChunk is the number of 16-sample groups dlyAccErrAvx2 runs
// through its 256-bit pass before it adds up their sums.
const dlyAccErrChunk = 32

// dlyAccErrAvx2 alternates three passes per chunk of dlyAccErrChunk groups:
// the 256-bit pass accumulates x*x and stores each group's four sub-group
// sums in the order the running sum adds them (hadd lanes [0], [4], [1],
// [5]); after ClearAVXUpperBits the scalar pass runs the running sum and
// stores every error s - y; the 128-bit pass then adds the squared errors
// into accErr with fused multiply-adds. The passes communicate through
// buffers written long before they are read, so no load waits on a narrower
// store.
func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	if !dlyUseFma {
		return dlyAccErrAvx2Ref(h, x, y, accErr, x2)
	}
	groups := len(h) / 16
	n := 16 * groups
	h = h[:n:n]
	x = x[:n:n]
	accErr = accErr[: 4*groups : 4*groups]
	var s float32
	var sums, es [4 * dlyAccErrChunk]float32
	for g0 := 0; g0 < groups; g0 += dlyAccErrChunk {
		m := min(dlyAccErrChunk, groups-g0)
		hc := h[16*g0 : 16*(g0+m)]
		xc := x[16*g0 : 16*(g0+m) : 16*(g0+m)]
		x2v := dlyLd8((*[8]float32)(x2[0:8]))
		x2v8 := dlyLd8((*[8]float32)(x2[8:16]))
		for g := 0; g < m && 16*g < len(hc)-15; g++ {
			i := 16 * g
			hh := (*[16]float32)(hc[i : i+16])
			xx := (*[16]float32)(xc[i : i+16])
			xk := dlyLd8((*[8]float32)(xx[0:8]))
			xk8 := dlyLd8((*[8]float32)(xx[8:16]))
			p := dlyLd8((*[8]float32)(hh[0:8])).Mul(xk)
			q := dlyLd8((*[8]float32)(hh[8:16])).Mul(xk8)
			x2v = xk.MulAdd(xk, x2v)
			x2v8 = xk8.MulAdd(xk8, x2v8)
			r := p.ConcatAddPairsGrouped(q)
			r = r.ConcatAddPairsGrouped(r)
			r.GetLo().ConcatPermuteScalars(0, 4, 1, 5, r.GetHi()).StoreArray((*[4]float32)(sums[4*g : 4*g+4]))
		}
		x2v.StoreArray((*[8]float32)(x2[0:8]))
		x2v8.StoreArray((*[8]float32)(x2[8:16]))
		archsimd.ClearAVXUpperBits()
		sc := sums[:4*m]
		ec := es[:len(sc)]
		for j, v := range sc {
			s += v
			ec[j] = s - y
		}
		ac := accErr[4*g0 : 4*(g0+m)]
		for j := 0; j < len(ac)-3 && j < len(ec)-3; j += 4 {
			ap := (*[4]float32)(ac[j : j+4])
			ev := dlyLd4((*[4]float32)(ec[j : j+4]))
			ev.MulAdd(ev, dlyLd4(ap)).StoreArray(ap)
		}
	}
	return s
}

func dlyAxpyFma(h, x []float32, alpha float32) {
	if !dlyUseFma {
		dlyAxpyFmaRef(h, x, alpha)
		return
	}
	n := len(h)
	x = x[:n:n]
	a := archsimd.BroadcastFloat32x8(alpha)
	i := 0
	for ; i < n-7; i += 8 {
		hp := (*[8]float32)(h[i : i+8])
		dlyLd8((*[8]float32)(x[i:i+8])).MulAdd(a, dlyLd8(hp)).StoreArray(hp)
	}
	archsimd.ClearAVXUpperBits()
	for ; i < n; i++ {
		h[i] = fmaf(x[i], alpha, h[i])
	}
}

var dlyPeakSimd = dlyUseAvx
