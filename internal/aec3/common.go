// Package aec3 is a Go port of WebRTC's AEC3 echo canceller
// (modules/audio_processing/aec3, WebRTC m141, branch-heads/7390).
//
// Naming mirrors the C++ sources: each C++ class keeps its name as a Go type,
// constructors are NewX, methods keep their C++ names, and kFoo constants drop
// the k prefix. All signal-domain storage and arithmetic is float32 to match
// the C++ float build, and products that the C++ build rounds separately are
// wrapped in float32(...) so the Go compiler never contracts them into FMA
// (libwebrtc is built with -ffp-contract=off).
package aec3

import (
	"math"
	"os"
)

// Aec3Optimization selects the kernel family, as in aec3_common.h.
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
	ExtendedBlockSize                        = 2 * FftLengthBy2
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

// DetectOptimization mirrors aec3_common.cc: AVX2, else SSE2 on amd64; NEON
// on arm64 (libwebrtc's arm64 build always has NEON); OptNone elsewhere. The
// purego build always reports OptNone (the scalar C reference lane).
//
// GOAEC3_FORCE_LANE=none|sse2|avx2|neon overrides the detected lane for
// testing (for example the AVX2 lane under Rosetta, which hides AVX/BMI2 from
// CPUID). A lane the architecture cannot run is ignored. The purego build
// ignores the override.
func DetectOptimization() Aec3Optimization {
	if purego {
		return OptNone
	}
	if opt, ok := forcedLane(); ok {
		return opt
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

func forcedLane() (Aec3Optimization, bool) {
	switch os.Getenv("GOAEC3_FORCE_LANE") {
	case "none":
		return OptNone, true
	case "sse2":
		return OptSse2, isAMD64
	case "avx2":
		return OptAvx2, isAMD64
	case "neon":
		return OptNeon, isARM64
	}
	return OptNone, false
}

// FastApproxLog2f reinterprets the float bits as an integer and rescales.
func FastApproxLog2f(in float32) float32 {
	out := float32(math.Float32bits(in))
	out = float32(out * 1.1920929e-7) // 1/2^23
	out -= 126.942695                 // Remove bias.
	return out
}

// Log2TodB converts a log2 value to dB (the C++ promotes through double).
func Log2TodB(inLog2 float32) float32 {
	return float32(3.0102999566398121 * float64(inLog2))
}

// Float minimum, maximum, clamp and absolute value with the exact semantics
// of the C++ expressions they port. Go's builtin min/max return NaN when any
// argument is NaN and order the zeros (min(-0, +0) = -0, max(-0, +0) = +0).
// std::min and std::max are single comparisons that return the first
// argument unless the second one compares smaller (larger): a NaN second
// argument is dropped, a NaN first argument is kept, and equal arguments
// (-0 and +0) give the first. The SIMD lanes have their own semantics: NEON
// vminq_f32/vmaxq_f32 are Go's builtin min/max (NaN propagates; they differ
// from Go only in which NaN payload they return), and SSE2/AVX2
// _mm_max_ps(a, b) is sseMax.

// stdMin is std::min(a, b) on floats: b < a ? b : a.
func stdMin(a, b float32) float32 {
	if b < a {
		return b
	}
	return a
}

// stdMax is std::max(a, b) on floats: a < b ? b : a.
func stdMax(a, b float32) float32 {
	if a < b {
		return b
	}
	return a
}

// stSafeClamp is webrtc::SafeClamp on floats:
// x <= lo ? lo : x >= hi ? hi : x.
func stSafeClamp(x, lo, hi float32) float32 {
	if x <= lo {
		return lo
	}
	if x >= hi {
		return hi
	}
	return x
}

// sseMax is _mm_max_ps(a, b) and _mm256_max_ps(a, b) per lane:
// a > b ? a : b, which gives b when either is NaN or both are zero.
func sseMax(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// fabsf is fabsf and std::fabs: x with the sign bit cleared.
func fabsf(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}

// stdMinmaxElement returns the values at the iterators std::minmax_element
// returns for x (len(x) > 0), with the pairwise algorithm of libc++. Without
// NaN that is the first minimum and the last maximum; with NaN the result
// depends on the pairing, which this follows comparison for comparison.
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
