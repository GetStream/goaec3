package aec3

// SubtractorOutputAnalyzer mirrors subtractor_output_analyzer.h: analyzes
// the properties of the subtractor output.
type SubtractorOutputAnalyzer struct {
	*subtractorOutputAnalyzerState
	more *soaMore // with more than inlineChannels channels
}

// subtractorOutputAnalyzerState is the state of a SubtractorOutputAnalyzer
// but its pointer.
type subtractorOutputAnalyzerState struct {
	numCaptureChannels int
	filtersConverged   [inlineChannels]bool
}

// soaMore is the per-channel state of a SubtractorOutputAnalyzer with more
// than inlineChannels capture channels.
type soaMore struct {
	filtersConverged []bool
}

// bind makes s the view of st.
func (s *SubtractorOutputAnalyzer) bind(st *subtractorOutputAnalyzerState, b *binder) {
	s.subtractorOutputAnalyzerState = st
	if b != nil {
		bindPtr(b, &s.more, &b.x.soaMore)
	}
}

func NewSubtractorOutputAnalyzer(numCaptureChannels int) *SubtractorOutputAnalyzer {
	s := &SubtractorOutputAnalyzer{}
	s.bind(&subtractorOutputAnalyzerState{}, nil)
	s.init(numCaptureChannels)
	return s
}

// init sets up s in place.
func (s *SubtractorOutputAnalyzer) init(numCaptureChannels int) {
	st := s.subtractorOutputAnalyzerState
	*st = subtractorOutputAnalyzerState{numCaptureChannels: numCaptureChannels}
	*s = SubtractorOutputAnalyzer{}
	s.bind(st, nil)
	if numCaptureChannels > inlineChannelLimit {
		s.more = &soaMore{filtersConverged: make([]bool, numCaptureChannels)}
	}
}

// Update analyzes the subtractor output.
func (s *SubtractorOutputAnalyzer) Update(subtractorOutput []SubtractorOutput) (anyFilterConverged, anyCoarseFilterConverged, allFiltersDiverged bool) {
	allFiltersDiverged = true
	filtersConverged := s.ConvergedFilters()
	for ch := range subtractorOutput {
		y2 := subtractorOutput[ch].Y2
		e2Refined := subtractorOutput[ch].E2RefinedSum
		e2Coarse := subtractorOutput[ch].E2CoarseSum
		const convergenceThreshold float32 = 50 * 50 * BlockSize
		const convergenceThresholdLowLevel float32 = 20 * 20 * BlockSize
		refinedFilterConverged := e2Refined < 0.5*y2 && y2 > convergenceThreshold
		coarseFilterConvergedStrict := e2Coarse < 0.05*y2 && y2 > convergenceThreshold
		coarseFilterConvergedRelaxed := e2Coarse < 0.3*y2 && y2 > convergenceThresholdLowLevel
		minE2 := stdMin(e2Refined, e2Coarse)
		filterDiverged := minE2 > 1.5*y2 && y2 > 30*30*BlockSize
		filtersConverged[ch] = refinedFilterConverged || coarseFilterConvergedStrict
		anyFilterConverged = anyFilterConverged || filtersConverged[ch]
		anyCoarseFilterConverged = anyCoarseFilterConverged || coarseFilterConvergedRelaxed
		allFiltersDiverged = allFiltersDiverged && filterDiverged
	}
	return
}

// ConvergedFilters returns the per-channel convergence flags.
func (s *SubtractorOutputAnalyzer) ConvergedFilters() []bool {
	if s.more != nil {
		return s.more.filtersConverged
	}
	return s.filtersConverged[:s.numCaptureChannels:s.numCaptureChannels]
}

// HandleEchoPathChange clears the convergence flags.
func (s *SubtractorOutputAnalyzer) HandleEchoPathChange() { clear(s.ConvergedFilters()) }
