//go:build goexperiment.simd && amd64 && !purego

package bands

import "simd/archsimd"

// archsimd FloatS16 conversions for amd64 (AVX, 128-bit). VMINPS/VMAXPS
// return their second operand on NaN and the compiler treats them as
// commutative, so the clamps are written as compare-and-select: a NaN fails
// both compares and passes through, as with Go's min/max, and the following
// multiply or int conversion sees the same value as the scalar reference.
// The scalar float-to-int16 conversion truncates to int32 (NaN becomes
// 0x80000000) and keeps the low 16 bits, so NaN gives 0; the kernel zeroes
// NaN lanes before converting. After the clamp every value fits int16, so
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
		// v + copysign(0.5, v), with NaN lanes zeroed.
		a = a.Add(a.AsUint32x4().And(sign).Or(half).AsFloat32x4()).Masked(a.Equal(a))
		b = b.Add(b.AsUint32x4().And(sign).Or(half).AsFloat32x4()).Masked(b.Equal(b))
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
	hi := archsimd.BroadcastFloat32x4(1)
	lo := archsimd.BroadcastFloat32x4(-1)
	scale := archsimd.BroadcastFloat32x4(32768)
	i := 0
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
	hi := archsimd.BroadcastFloat32x4(32768)
	lo := archsimd.BroadcastFloat32x4(-32768)
	scale := archsimd.BroadcastFloat32x4(1.0 / 32768)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat32x4Array((*[4]float32)(src[i : i+4]))
		bsClamp4(v, hi, lo).Mul(scale).StoreArray((*[4]float32)(dest[i : i+4]))
	}
	floatS16ToFloatSliceRef(src[i:], dest[i:])
}
