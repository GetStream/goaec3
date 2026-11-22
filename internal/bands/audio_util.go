// Package bands implements the band splitting WebRTC's AudioBuffer applies
// around AEC3: the fixed-point two-band QMF for 32 kHz
// (common_audio/signal_processing/splitting_filter.c), the float three-band
// filter bank for 48 kHz (modules/audio_processing/three_band_filter_bank.cc),
// the SplittingFilter that dispatches to them, and the conversions to and
// from FloatS16, a float32 in [-32768, 32768]. Every function is bit-exact
// with its C++ counterpart.
package bands

import "math"

// S16ToFloat converts an int16 sample to [-1, 1].
func S16ToFloat(v int16) float32 {
	const scaling float32 = 1.0 / 32768
	return float32(v) * scaling
}

// The clamps are std::min(v, hi) then std::max(v, lo) in the C++. Go's min
// and max agree with them: the bounds are nonzero, so no two zeros tie, and
// a NaN v stays NaN in both.

// FloatS16ToS16 clamps and rounds a FloatS16 sample to int16.
func FloatS16ToS16(v float32) int16 {
	v = min(v, 32767)
	v = max(v, -32768)
	return int16(v + bsCopysignHalf(v))
}

// FloatToS16 scales a [-1, 1] sample to int16 with rounding and clamping.
func FloatToS16(v float32) int16 {
	v *= 32768
	v = min(v, 32767)
	v = max(v, -32768)
	return int16(v + bsCopysignHalf(v))
}

// FloatToFloatS16 clamps v to [-1, 1] and scales it to FloatS16.
func FloatToFloatS16(v float32) float32 {
	v = min(v, 1)
	v = max(v, -1)
	return v * 32768
}

// FloatS16ToFloat clamps v to [-32768, 32768] and scales it to [-1, 1].
func FloatS16ToFloat(v float32) float32 {
	v = min(v, 32768)
	v = max(v, -32768)
	const scaling float32 = 1.0 / 32768
	return v * scaling
}

func bsCopysignHalf(v float32) float32 {
	if math.Float32bits(v)&(1<<31) != 0 {
		return -0.5
	}
	return 0.5
}

// The Ref functions are the scalar spec of the slice conversions, which
// are archsimd kernels under GOEXPERIMENT=simd (audio_util_simd_*.go).

// floatS16ToS16SliceRef is the slice form of FloatS16ToS16 in audio_util.cc.
// Converting NaN to int16 is undefined in C++, so a NaN sample becomes what
// the loop clang compiles makes of it: bsNaNS16Vector in the vector body,
// which takes eight samples at a time, and 0 in the scalar remainder.
func floatS16ToS16SliceRef(src []float32, dest []int16) {
	dest = dest[:len(src)]
	vec := len(src) &^ 7
	for i, v := range src {
		d := FloatS16ToS16(v)
		if v != v && i < vec {
			d = bsNaNS16Vector
		}
		dest[i] = d
	}
}

func s16ToFloatS16SliceRef(src []int16, dest []float32) {
	dest = dest[:len(src)]
	for i, v := range src {
		dest[i] = float32(v)
	}
}

func floatToFloatS16SliceRef(src, dest []float32) {
	dest = dest[:len(src)]
	for i, v := range src {
		dest[i] = FloatToFloatS16(v)
	}
}

func floatS16ToFloatSliceRef(src, dest []float32) {
	dest = dest[:len(src)]
	for i, v := range src {
		dest[i] = FloatS16ToFloat(v)
	}
}
