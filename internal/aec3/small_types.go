package aec3

import "math"

// DelayAdjustment mirrors EchoPathVariability::DelayAdjustment.
type DelayAdjustment int

const (
	DelayAdjustmentNone DelayAdjustment = iota
	DelayAdjustmentBufferFlush
	DelayAdjustmentNewDetectedDelay
)

// EchoPathVariability mirrors echo_path_variability.h.
type EchoPathVariability struct {
	GainChange  bool
	DelayChange DelayAdjustment
	ClockDrift  bool
}

func (e EchoPathVariability) AudioPathChanged() bool {
	return e.GainChange || e.DelayChange != DelayAdjustmentNone
}

// DelayEstimateQuality mirrors DelayEstimate::Quality.
type DelayEstimateQuality int

const (
	DelayQualityCoarse DelayEstimateQuality = iota
	DelayQualityRefined
)

// DelayEstimate mirrors delay_estimate.h.
type DelayEstimate struct {
	Quality               DelayEstimateQuality
	Delay                 int
	BlocksSinceLastChange int
	BlocksSinceLastUpdate int
}

// SubtractorOutput mirrors subtractor_output.h.
type SubtractorOutput struct {
	SRefined       [BlockSize]float32
	SCoarse        [BlockSize]float32
	ERefined       [BlockSize]float32
	ECoarse        [BlockSize]float32
	ERefinedFft    FftData // E_refined
	E2Refined      [FftLengthBy2Plus1]float32
	E2Coarse       [FftLengthBy2Plus1]float32
	S2Refined      float32
	S2Coarse       float32
	E2RefinedSum   float32 // e2_refined
	E2CoarseSum    float32 // e2_coarse
	Y2             float32
	SRefinedMaxAbs float32
	SCoarseMaxAbs  float32
}

func (s *SubtractorOutput) Reset() {
	s.SRefined = [BlockSize]float32{}
	s.SCoarse = [BlockSize]float32{}
	s.ERefined = [BlockSize]float32{}
	s.ECoarse = [BlockSize]float32{}
	s.ERefinedFft.Clear()
	s.E2Refined = [FftLengthBy2Plus1]float32{}
	s.E2Coarse = [FftLengthBy2Plus1]float32{}
	s.E2RefinedSum, s.E2CoarseSum = 0, 0
	s.S2Refined, s.S2Coarse = 0, 0
	s.Y2 = 0
}

// maxAbsOf is std::max(max, -min) of the largest and the smallest element of
// x, each found by a loop from x[0] with "if v > max" and "if v < min".
// Without a NaN and with an element that is not zero, that is the largest
// magnitude in x (kernMaxAbsBits); maxAbsOfRef runs the loops.
func maxAbsOf(x []float32) float32 {
	if mn, mx, ok := kernMinMax(x); ok {
		return stdMax(mx, -mn)
	}
	if m := kernMaxAbsBits(x); m-1 < kernInfBits {
		return math.Float32frombits(m)
	}
	return maxAbsOfRef(x)
}

func maxAbsOfRef(x []float32) float32 {
	mx, mn := x[0], x[0]
	for _, v := range x[1:] {
		if v > mx {
			mx = v
		}
		if v < mn {
			mn = v
		}
	}
	return stdMax(mx, -mn)
}

func (s *SubtractorOutput) ComputeMetrics(y []float32) {
	s.Y2, s.E2RefinedSum, s.E2CoarseSum, s.S2Refined, s.S2Coarse = kernSumSq5(
		(*[BlockSize]float32)(y), &s.ERefined, &s.ECoarse, &s.SRefined, &s.SCoarse)
	s.SRefinedMaxAbs = maxAbsOf(s.SRefined[:])
	s.SCoarseMaxAbs = maxAbsOf(s.SCoarse[:])
}
