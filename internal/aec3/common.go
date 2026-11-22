// Package aec3 is a Go port of the AEC3 echo canceller of WebRTC m141
// (modules/audio_processing/aec3, branch-heads/7390), bit-exact with the C++
// in every lane.
//
// A C++ class keeps its name as a Go type, its constructor is NewX, methods
// keep their names and kFoo constants drop the k. Signal arithmetic is
// float32. libwebrtc is built with -ffp-contract=off, so a product that
// feeds a sum is wrapped in float32(...), which keeps the compiler from
// fusing the two.
package aec3

import (
	"math"
	"sync/atomic"
)

// Aec3Optimization selects the lane of the kernels, as in aec3_common.h.
type Aec3Optimization int

const (
	OptNone Aec3Optimization = iota
	OptSse2
	OptAvx2
	OptNeon
)

const (
	NumBlocksPerSecond                       = 250
	MetricsReportingIntervalBlocks           = 10 * NumBlocksPerSecond
	MetricsComputationBlocks                 = 3
	MetricsCollectionBlocks                  = MetricsReportingIntervalBlocks - MetricsComputationBlocks
	FftLengthBy2                             = 64
	FftLengthBy2Plus1                        = FftLengthBy2 + 1
	FftLengthBy2Minus1                       = FftLengthBy2 - 1
	FftLength                                = 2 * FftLengthBy2
	FftLengthBy2Log2                         = 6
	RenderTransferQueueSizeFrames            = 100
	MaxNumBands                              = 3
	FrameSize                                = 160
	SubFrameLength                           = FrameSize / 2
	BlockSize                                = FftLengthBy2
	BlockSizeLog2                            = FftLengthBy2Log2
	MatchedFilterWindowSizeSubBlocks         = 32
	MatchedFilterAlignmentShiftSizeSubBlocks = MatchedFilterWindowSizeSubBlocks * 3 / 4
)

func NumBandsForRate(sampleRateHz int) int { return sampleRateHz / 16000 }

func ValidFullBandRate(sampleRateHz int) bool {
	return sampleRateHz == 16000 || sampleRateHz == 32000 || sampleRateHz == 48000
}

func GetTimeDomainLength(filterLengthBlocks int) int { return filterLengthBlocks * FftLengthBy2 }

func GetDownSampledBufferSize(downSamplingFactor, numMatchedFilters int) int {
	return BlockSize / downSamplingFactor *
		(MatchedFilterAlignmentShiftSizeSubBlocks*numMatchedFilters + MatchedFilterWindowSizeSubBlocks + 1)
}

func GetRenderDelayBufferSize(downSamplingFactor, numMatchedFilters, filterLengthBlocks int) int {
	return GetDownSampledBufferSize(downSamplingFactor, numMatchedFilters)/(BlockSize/downSamplingFactor) +
		filterLengthBlocks + 1
}

// DetectOptimization returns the lane libwebrtc selects: AVX2 or SSE2 on
// amd64, NEON on arm64, none elsewhere and in the purego build. A lane set
// with SetForcedLaneForTesting replaces it.
func DetectOptimization() Aec3Optimization {
	if purego {
		return OptNone
	}
	if v := forcedLane.Load(); v != 0 {
		return Aec3Optimization(v - 1)
	}
	switch {
	case isAMD64 && hasAVX2():
		return OptAvx2
	case isAMD64:
		return OptSse2
	case isARM64:
		return OptNeon
	}
	return OptNone
}

// forcedLane is the lane SetForcedLaneForTesting set, plus one, or zero.
var forcedLane atomic.Int32

// SetForcedLaneForTesting makes DetectOptimization return opt. It reports
// false and does nothing if the architecture cannot run opt. The purego
// build ignores the forced lane.
func SetForcedLaneForTesting(opt Aec3Optimization) bool {
	switch {
	case opt == OptNone, opt == OptSse2 && isAMD64, opt == OptAvx2 && isAMD64, opt == OptNeon && isARM64:
		forcedLane.Store(int32(opt) + 1)
		return true
	}
	return false
}

// ClearForcedLaneForTesting restores the detected lane.
func ClearForcedLaneForTesting() { forcedLane.Store(0) }

// FastApproxLog2f approximates log2(in) from the bits of in.
func FastApproxLog2f(in float32) float32 {
	out := float32(math.Float32bits(in))
	out = float32(out * 1.1920929e-7) // 1/2^23
	out -= 126.942695                 // Remove bias.
	return out
}

// Log2TodB converts a log2 value to dB. The product is computed in double,
// as in the C++.
func Log2TodB(inLog2 float32) float32 {
	return float32(3.0102999566398121 * float64(inLog2))
}

// Float minimum, maximum, clamp and absolute value as the C++ expressions
// compute them. std::min and std::max are one comparison and return the
// first argument unless the second is smaller (larger): they drop a NaN
// second argument, keep a NaN first argument, and return the first of -0 and
// +0. Go's min and max return NaN for a NaN argument and order the zeros; so
// do NEON vminq_f32 and vmaxq_f32, up to the payload of the NaN. _mm_max_ps
// is sseMax.

// stdMin is std::min(a, b): b < a ? b : a.
func stdMin(a, b float32) float32 {
	if b < a {
		return b
	}
	return a
}

// stdMax is std::max(a, b): a < b ? b : a.
func stdMax(a, b float32) float32 {
	if a < b {
		return b
	}
	return a
}

// stSafeClamp is webrtc::SafeClamp: x <= lo ? lo : x >= hi ? hi : x.
func stSafeClamp(x, lo, hi float32) float32 {
	if x <= lo {
		return lo
	}
	if x >= hi {
		return hi
	}
	return x
}

// sseMax is _mm_max_ps(a, b) per lane: a > b ? a : b. It returns b when
// either is NaN or both are zero.
func sseMax(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// fabsf returns x with the sign bit cleared.
func fabsf(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

// stdMinmaxElement returns the values std::minmax_element of libc++ points
// at for a non-empty x. Without a NaN they are the first minimum and the
// last maximum; with one they depend on the pairing of the comparisons,
// which this follows.
func stdMinmaxElement(x []float32) (mn, mx float32) {
	mn, mx = x[0], x[0]
	if len(x) == 1 {
		return mn, mx
	}
	if x[1] < mn {
		mn = x[1]
	} else {
		mx = x[1]
	}
	i := 2
	for ; i+1 < len(x); i += 2 {
		a, b := x[i], x[i+1]
		if b < a {
			if b < mn {
				mn = b
			}
			if !(a < mx) {
				mx = a
			}
		} else {
			if a < mn {
				mn = a
			}
			if !(b < mx) {
				mx = b
			}
		}
	}
	if i < len(x) {
		if a := x[i]; a < mn {
			mn = a
		} else if !(a < mx) {
			mx = a
		}
	}
	return mn, mx
}
