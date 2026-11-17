package aec3

// SubtractorOutputAnalyzer mirrors subtractor_output_analyzer.h: analyzes
// the properties of the subtractor output.
type SubtractorOutputAnalyzer struct {
	filtersConverged []bool
}

func NewSubtractorOutputAnalyzer(numCaptureChannels int) *SubtractorOutputAnalyzer {
	return &SubtractorOutputAnalyzer{filtersConverged: make([]bool, numCaptureChannels)}
}

// Update analyzes the subtractor output.
func (s *SubtractorOutputAnalyzer) Update(subtractorOutput []SubtractorOutput) (anyFilterConverged, anyCoarseFilterConverged, allFiltersDiverged bool) {
	allFiltersDiverged = true
	for ch := range subtractorOutput {
		y2 := subtractorOutput[ch].Y2
		e2Refined := subtractorOutput[ch].E2RefinedSum
		e2Coarse := subtractorOutput[ch].E2CoarseSum
		const convergenceThreshold float32 = 50 * 50 * BlockSize
		const convergenceThresholdLowLevel float32 = 20 * 20 * BlockSize
		refinedFilterConverged := e2Refined < 0.5*y2 && y2 > convergenceThreshold
		coarseFilterConvergedStrict := e2Coarse < 0.05*y2 && y2 > convergenceThreshold
		coarseFilterConvergedRelaxed := e2Coarse < 0.3*y2 && y2 > convergenceThresholdLowLevel
		minE2 := stMinf(e2Refined, e2Coarse)
		filterDiverged := minE2 > 1.5*y2 && y2 > 30*30*BlockSize
		s.filtersConverged[ch] = refinedFilterConverged || coarseFilterConvergedStrict
		anyFilterConverged = anyFilterConverged || s.filtersConverged[ch]
		anyCoarseFilterConverged = anyCoarseFilterConverged || coarseFilterConvergedRelaxed
		allFiltersDiverged = allFiltersDiverged && filterDiverged
	}
	return
}

// ConvergedFilters returns the per-channel convergence flags.
func (s *SubtractorOutputAnalyzer) ConvergedFilters() []bool { return s.filtersConverged }

// HandleEchoPathChange clears the convergence flags.
func (s *SubtractorOutputAnalyzer) HandleEchoPathChange() { clear(s.filtersConverged) }
