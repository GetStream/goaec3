//go:build goexperiment.simd && (arm64 || amd64) && !purego

package aec3

import "simd/archsimd"

// dlyMaxSquarePeakIndex computes dlyMaxSquarePeakIndexRef with sixteen lanes.
//
// The reference scans even and odd indices separately with strict
// comparisons, starting from h[0]² and h[1]². When neither start is NaN, a
// NaN square never wins and never becomes the maximum, so each scan returns
// the earliest index of the largest non-NaN square. Lane l tracks the same
// for the indices ≡ l (mod 16) with strict comparisons from a -1 sentinel
// (squares are >= 0); the even and odd lanes then merge to the even and
// odd results, taking the earlier index on ties, and the scalar tail
// continues both scans. A NaN start falls back to the reference.
func dlyMaxSquarePeakIndex(h []float32) int {
	if len(h) < 16 || !dlyPeakSimd || h[0]*h[0] != h[0]*h[0] || h[1]*h[1] != h[1]*h[1] {
		return dlyMaxSquarePeakIndexRef(h)
	}
	covered := len(h) &^ 1
	m := covered &^ 15
	var mv [16]float32
	var iv [16]int32
	p := h[:m]
	if !dlyPeakLanes256(p, &mv, &iv) {
		dlyPeakLanes128(p, &mv, &iv)
	}
	maxElement1, lagEstimate1 := mv[0], int(iv[0])
	maxElement2, lagEstimate2 := mv[1], int(iv[1])
	for l := 2; l < 16; l += 2 {
		if v, i := mv[l], int(iv[l]); v > maxElement1 || (v == maxElement1 && i < lagEstimate1) {
			maxElement1, lagEstimate1 = v, i
		}
		if v, i := mv[l+1], int(iv[l+1]); v > maxElement2 || (v == maxElement2 && i < lagEstimate2) {
			maxElement2, lagEstimate2 = v, i
		}
	}
	for k := m; k < covered; k += 2 {
		element1 := h[k] * h[k]
		element2 := h[k+1] * h[k+1]
		if element1 > maxElement1 {
			maxElement1 = element1
			lagEstimate1 = k
		}
		if element2 > maxElement2 {
			maxElement2 = element2
			lagEstimate2 = k + 1
		}
	}
	if maxElement2 > maxElement1 {
		maxElement1 = maxElement2
		lagEstimate1 = lagEstimate2
	}
	lastIndex := len(h) - 1
	lastElement := h[lastIndex] * h[lastIndex]
	if lastElement > maxElement1 {
		return lastIndex
	}
	return lagEstimate1
}

// dlyPeakLanes128 sets mv[l] and iv[l] to the largest square, and its
// earliest index, among p[k] with k = l (mod 16), from a -1 sentinel.
func dlyPeakLanes128(p []float32, mv *[16]float32, iv *[16]int32) {
	m0 := archsimd.BroadcastFloat32x4(-1)
	m1, m2, m3 := m0, m0, m0
	i0 := archsimd.BroadcastInt32x4(-1)
	i1, i2, i3 := i0, i0, i0
	cur := archsimd.LoadInt32x4Array(&[4]int32{0, 1, 2, 3})
	step := archsimd.BroadcastInt32x4(4)
	for len(p) >= 16 {
		pp := (*[16]float32)(p)
		v0 := archsimd.LoadFloat32x4Array((*[4]float32)(pp[0:4]))
		v1 := archsimd.LoadFloat32x4Array((*[4]float32)(pp[4:8]))
		v2 := archsimd.LoadFloat32x4Array((*[4]float32)(pp[8:12]))
		v3 := archsimd.LoadFloat32x4Array((*[4]float32)(pp[12:16]))
		v0, v1, v2, v3 = v0.Mul(v0), v1.Mul(v1), v2.Mul(v2), v3.Mul(v3)
		c1 := cur.Add(step)
		c2 := c1.Add(step)
		c3 := c2.Add(step)
		g0, g1, g2, g3 := v0.Greater(m0), v1.Greater(m1), v2.Greater(m2), v3.Greater(m3)
		m0, i0 = v0.IfElse(g0, m0), cur.IfElse(g0, i0)
		m1, i1 = v1.IfElse(g1, m1), c1.IfElse(g1, i1)
		m2, i2 = v2.IfElse(g2, m2), c2.IfElse(g2, i2)
		m3, i3 = v3.IfElse(g3, m3), c3.IfElse(g3, i3)
		cur = c3.Add(step)
		p = p[16:]
	}
	m0.StoreArray((*[4]float32)(mv[0:4]))
	m1.StoreArray((*[4]float32)(mv[4:8]))
	m2.StoreArray((*[4]float32)(mv[8:12]))
	m3.StoreArray((*[4]float32)(mv[12:16]))
	i0.StoreArray((*[4]int32)(iv[0:4]))
	i1.StoreArray((*[4]int32)(iv[4:8]))
	i2.StoreArray((*[4]int32)(iv[8:12]))
	i3.StoreArray((*[4]int32)(iv[12:16]))
}
