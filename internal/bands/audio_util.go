// Package bands ports the WebRTC band-splitting filters used around AEC3:
// the 2-band fixed-point QMF for 32 kHz (common_audio/signal_processing/
// splitting_filter.c), the float 3-band filter bank for 48 kHz
// (modules/audio_processing/three_band_filter_bank.cc), the SplittingFilter
// dispatcher and the FloatS16 conversions AudioBuffer applies around them.
//
// The FloatS16 representation is a float32 in [-32768, 32768].
package bands

import "math"

// S16ToFloat converts an int16 sample to [-1, 1].
func S16ToFloat(v int16) float32 {
	const scaling float32 = 1.0 / 32768
	return float32(v) * scaling
}

// The clamps below are std::min(v, hi) then std::max(v, lo) in the C++.
// With nonzero bounds Go's min/max give the same results: no tie between
// zeros is possible, and a NaN v gives NaN either way (std::min/std::max
// return v itself; the bound is never NaN).

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

// The slice conversions dispatch to archsimd kernels under GOEXPERIMENT=simd
// (audio_util_simd_*.go). The ...Ref functions are the scalar spec: every
// kernel is elementwise and bit-identical to them.

// floatS16ToS16SliceRef is the FloatS16ToS16(const float*, size_t, int16_t*)
// loop of audio_util.cc. Converting NaN to int16 is undefined in C++, so the
// result for a NaN sample is what the compiled loop does: clang vectorizes
// it eight samples at a time, the scalar remainder converts through int32
// and keeps the low 16 bits (NaN gives 0 on x86 and arm64), and the vector
// body gives bsNaNS16Vector (x86: cvttps2dq gives 0x80000000, which
// packssdw saturates to -32768; arm64: fcvtzs gives 0).
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
