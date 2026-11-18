//go:build goexperiment.simd && arm64 && !purego

package bands

import "simd/archsimd"

// archsimd FloatS16 conversions for arm64. Go's scalar min/max compile to
// FMIN/FMAX, the instructions Float32x4.Min/Max emit, and the clamp bounds
// are never NaN or zero, so each lane equals the scalar reference. The
// float-to-int16 conversion truncates with FCVTZS like the scalar code (NaN
// becomes 0); after the clamp every value fits int16, so the narrowing is
// exact.

// FloatS16ToS16Slice converts src into dest (len(src) samples).
func FloatS16ToS16Slice(src []float32, dest []int16) {
	n := len(src)
	dest = dest[:n]
	hi := archsimd.BroadcastFloat32x4(32767)
	lo := archsimd.BroadcastFloat32x4(-32768)
	sign := archsimd.BroadcastUint32x4(1 << 31)
	half := archsimd.BroadcastUint32x4(0x3f000000) // 0.5
	i := 0
	for ; i+8 <= n; i += 8 {
		a := archsimd.LoadFloat32x4Array((*[4]float32)(src[i : i+4])).Min(hi).Max(lo)
		b := archsimd.LoadFloat32x4Array((*[4]float32)(src[i+4 : i+8])).Min(hi).Max(lo)
		// v + copysign(0.5, v).
		a = a.Add(a.ToBits().And(sign).Or(half).BitsToFloat32())
		b = b.Add(b.ToBits().And(sign).Or(half).BitsToFloat32())
		ai := a.ConvertToInt32().ToBits().ReshapeToUint16s()
		bi := b.ConvertToInt32().ToBits().ReshapeToUint16s()
		ai.ConcatEven(bi).BitsToInt16().StoreArray((*[8]int16)(dest[i : i+8]))
	}
	floatS16ToS16SliceRef(src[i:], dest[i:])
}

// S16ToFloatS16Slice converts src into dest (len(src) samples).
func S16ToFloatS16Slice(src []int16, dest []float32) {
	n := len(src)
	dest = dest[:n]
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadInt16x8Array((*[8]int16)(src[i : i+8]))
		v.ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(dest[i : i+4]))
		v.HiToLo().ExtendLo4ToInt32().ConvertToFloat32().StoreArray((*[4]float32)(dest[i+4 : i+8]))
	}
	s16ToFloatS16SliceRef(src[i:], dest[i:])
}

// FloatToFloatS16Slice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatToFloatS16Slice(src, dest []float32) {
	n := len(src)
	dest = dest[:n]
	hi := archsimd.BroadcastFloat32x4(1)
	lo := archsimd.BroadcastFloat32x4(-1)
	scale := archsimd.BroadcastFloat32x4(32768)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat32x4Array((*[4]float32)(src[i : i+4]))
		v.Min(hi).Max(lo).Mul(scale).StoreArray((*[4]float32)(dest[i : i+4]))
	}
	floatToFloatS16SliceRef(src[i:], dest[i:])
}

// FloatS16ToFloatSlice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatS16ToFloatSlice(src, dest []float32) {
	n := len(src)
	dest = dest[:n]
	hi := archsimd.BroadcastFloat32x4(32768)
	lo := archsimd.BroadcastFloat32x4(-32768)
	scale := archsimd.BroadcastFloat32x4(1.0 / 32768)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat32x4Array((*[4]float32)(src[i : i+4]))
		v.Min(hi).Max(lo).Mul(scale).StoreArray((*[4]float32)(dest[i : i+4]))
	}
	floatS16ToFloatSliceRef(src[i:], dest[i:])
}
