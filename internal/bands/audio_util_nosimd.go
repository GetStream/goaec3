//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package bands

// FloatS16ToS16Slice converts src into dest (len(src) samples).
func FloatS16ToS16Slice(src []float32, dest []int16) { floatS16ToS16SliceRef(src, dest) }

// S16ToFloatS16Slice converts src into dest (len(src) samples).
func S16ToFloatS16Slice(src []int16, dest []float32) { s16ToFloatS16SliceRef(src, dest) }

// FloatToFloatS16Slice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatToFloatS16Slice(src, dest []float32) { floatToFloatS16SliceRef(src, dest) }

// FloatS16ToFloatSlice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatS16ToFloatSlice(src, dest []float32) { floatS16ToFloatSliceRef(src, dest) }
