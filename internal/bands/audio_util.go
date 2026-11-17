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

// FloatS16ToS16Slice converts src into dest (len(src) samples).
func FloatS16ToS16Slice(src []float32, dest []int16) {
	dest = dest[:len(src)]
	for i, v := range src {
		dest[i] = FloatS16ToS16(v)
	}
}

// S16ToFloatS16Slice converts src into dest (len(src) samples).
func S16ToFloatS16Slice(src []int16, dest []float32) {
	dest = dest[:len(src)]
	for i, v := range src {
		dest[i] = float32(v)
	}
}

// FloatToFloatS16Slice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatToFloatS16Slice(src, dest []float32) {
	dest = dest[:len(src)]
	for i, v := range src {
		dest[i] = FloatToFloatS16(v)
	}
}

// FloatS16ToFloatSlice converts src into dest (len(src) samples); src and
// dest may alias.
func FloatS16ToFloatSlice(src, dest []float32) {
	dest = dest[:len(src)]
	for i, v := range src {
		dest[i] = FloatS16ToFloat(v)
	}
}
