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
func DetectOptimization() Aec3Optimization {
	switch {
	case purego:
		return OptNone
	case isAMD64 && hasAVX2():
		return OptAvx2
	case isAMD64:
		return OptSse2
	case isARM64:
		return OptNeon
	}
	return OptNone
}

// FastApproxLog2f reinterprets the float bits as an integer and rescales.
func FastApproxLog2f(in float32) float32 {
	out := float32(math.Float32bits(in))
	out *= 1.1920929e-7 // 1/2^23
	out -= 126.942695   // Remove bias.
	return out
}

// Log2TodB converts a log2 value to dB (the C++ promotes through double).
func Log2TodB(inLog2 float32) float32 {
	return float32(3.0102999566398121 * float64(inLog2))
}
