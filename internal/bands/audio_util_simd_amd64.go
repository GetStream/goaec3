//go:build goexperiment.simd && amd64 && !purego

package bands

import "simd/archsimd"

// archsimd FloatS16 conversions for amd64 (AVX, 128-bit). VMINPS/VMAXPS
// return their second operand on NaN and the compiler treats them as
// commutative, so the clamps are written as compare-and-select: a NaN fails
// both compares and passes through, as with Go's min/max, and the following
// multiply or int conversion sees the same value as the scalar reference.
// A NaN sample converts to -32768 in the vector body (bsNaNS16Vector) and
// to 0 in the scalar remainder, like the compiled C++ loop
// (floatS16ToS16SliceRef). After the clamp every value fits int16, so
// the saturating pack is exact.

// bsClamp4 is max(min(v, hi), lo) for the finite bounds hi > lo.
func bsClamp4(v, hi, lo archsimd.Float32x4) archsimd.Float32x4 {
	v = hi.IfElse(v.Greater(hi), v)
	return lo.IfElse(v.Less(lo), v)
}

// FloatS16ToS16Slice converts src into dest (len(src) samples).
func FloatS16ToS16Slice(src []float32, dest []int16) {
	if !bsHasSIMD {
		floatS16ToS16SliceRef(src, dest)
		return
	}
	n := len(src)
	dest = dest[:n]
	hi := archsimd.BroadcastFloat32x4(32767)
	lo := archsimd.BroadcastFloat32x4(-32768)
	sign := archsimd.BroadcastUint32x4(1 << 31)
	half := archsimd.BroadcastUint32x4(0x3f000000) // 0.5
	i := 0
	for ; i+8 <= n; i += 8 {
		a := bsClamp4(archsimd.LoadFloat32x4Array((*[4]float32)(src[i:i+4])), hi, lo)
		b := bsClamp4(archsimd.LoadFloat32x4Array((*[4]float32)(src[i+4:i+8])), hi, lo)
		// v + copysign(0.5, v); a NaN lane converts to 0x80000000 and
		// saturates to -32768, as in the vectorized C++ loop.
		a = a.Add(a.AsUint32x4().And(sign).Or(half).AsFloat32x4())
		b = b.Add(b.AsUint32x4().And(sign).Or(half).AsFloat32x4())
		a.ConvertToInt32().SaturateToInt16Concat(b.ConvertToInt32()).StoreArray((*[8]int16)(dest[i : i+8]))
	}
	floatS16ToS16SliceRef(src[i:], dest[i:])
}

// S16ToFloatS16Slice converts src into dest (len(src) samples).
func S16ToFloatS16Slice(src []int16, dest []float32) {
	if !bsHasSIMD {
		s16ToFloatS16SliceRef(src, dest)
		return
	}
	n := len(src)
	dest = dest[:n]
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(src[i : i+8]))
		v.ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(dest[i : i+4]))
		// Sign-extend the upper four: duplicate each and shift right
		// arithmetically by 16.
		v.InterleaveHi(v).AsInt32x4().ShiftAllRight(16).ConvertToFloat32().
			StoreArray((*[4]float32)(dest[i+4 : i+8]))
	}
	s16ToFloatS16SliceRef(src[i:], dest[i:])
}

// FloatToFloatS16Slice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatToFloatS16Slice(src, dest []float32) {
	if !bsHasSIMD {
		floatToFloatS16SliceRef(src, dest)
		return
	}
	n := len(src)
	dest = dest[:n]
	i := 0
	if bsHasAVX2 {
		i = bsClampScale256(src, dest, 1, -1, 32768)
	}
	hi := archsimd.BroadcastFloat32x4(1)
	lo := archsimd.BroadcastFloat32x4(-1)
	scale := archsimd.BroadcastFloat32x4(32768)
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat32x4Array((*[4]float32)(src[i : i+4]))
		bsClamp4(v, hi, lo).Mul(scale).StoreArray((*[4]float32)(dest[i : i+4]))
	}
	floatToFloatS16SliceRef(src[i:], dest[i:])
}

// FloatS16ToFloatSlice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatS16ToFloatSlice(src, dest []float32) {
	if !bsHasSIMD {
		floatS16ToFloatSliceRef(src, dest)
		return
	}
	n := len(src)
	dest = dest[:n]
	i := 0
	if bsHasAVX2 {
		i = bsClampScale256(src, dest, 32768, -32768, 1.0/32768)
	}
	hi := archsimd.BroadcastFloat32x4(32768)
	lo := archsimd.BroadcastFloat32x4(-32768)
	scale := archsimd.BroadcastFloat32x4(1.0 / 32768)
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat32x4Array((*[4]float32)(src[i : i+4]))
		bsClamp4(v, hi, lo).Mul(scale).StoreArray((*[4]float32)(dest[i : i+4]))
	}
	floatS16ToFloatSliceRef(src[i:], dest[i:])
}

// bsClamp8 is bsClamp4 on eight samples.
func bsClamp8(v, hi, lo archsimd.Float32x8) archsimd.Float32x8 {
	v = hi.IfElse(v.Greater(hi), v)
	return lo.IfElse(v.Less(lo), v)
}

// bsClampScale256 sets dest = max(min(src, hi), lo) * scale for the leading
// multiple of eight samples, sixteen per step, and returns their number. It
// is the 256-bit body of the two float conversions; it does not inline, so
// its VZEROUPPER comes before the caller's 128-bit and scalar remainder.
//
//go:noinline
func bsClampScale256(src, dest []float32, hi, lo, scale float32) int {
	dest = dest[:len(src)]
	hv := archsimd.BroadcastFloat32x8(hi)
	lv := archsimd.BroadcastFloat32x8(lo)
	sv := archsimd.BroadcastFloat32x8(scale)
	i := 0
	for ; i+16 <= len(src); i += 16 {
		p := (*[16]float32)(src[i : i+16])
		q := (*[16]float32)(dest[i : i+16])
		a := archsimd.LoadFloat32x8Array((*[8]float32)(p[0:8]))
		b := archsimd.LoadFloat32x8Array((*[8]float32)(p[8:16]))
		bsClamp8(a, hv, lv).Mul(sv).StoreArray((*[8]float32)(q[0:8]))
		bsClamp8(b, hv, lv).Mul(sv).StoreArray((*[8]float32)(q[8:16]))
	}
	for ; i+8 <= len(src); i += 8 {
		v := archsimd.LoadFloat32x8Array((*[8]float32)(src[i : i+8]))
		bsClamp8(v, hv, lv).Mul(sv).StoreArray((*[8]float32)(dest[i : i+8]))
	}
	archsimd.ClearAVXUpperBits()
	return i
}
