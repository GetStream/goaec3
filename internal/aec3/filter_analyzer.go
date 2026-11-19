package aec3

func stFindPeakIndex(filterTimeDomain []float32, peakIndexIn, startSample, endSample int) int {
	peakIndexOut := peakIndexIn
	maxH2 := filterTimeDomain[peakIndexOut] * filterTimeDomain[peakIndexOut]
	for k := startSample; k <= endSample; k++ {
		tmp := filterTimeDomain[k] * filterTimeDomain[k]
		if tmp > maxH2 {
			peakIndexOut = k
			maxH2 = tmp
		}
	}
	return peakIndexOut
}

// FilterRegion is the region of the filter analyzed in one block.
type FilterRegion struct {
	StartSample int
	EndSample   int
}

// stConsistentFilterDetector mirrors FilterAnalyzer::ConsistentFilterDetector.
type stConsistentFilterDetector struct {
	significantPeak           bool
	filterFloorAccum          float32
	filterSecondaryPeak       float32
	filterFloorLowLimit       int
	filterFloorHighLimit      int
	activeRenderThreshold     float32
	consistentEstimateCounter int
	consistentDelayReference  int
}

func (d *stConsistentFilterDetector) Reset() {
	d.significantPeak = false
	d.filterFloorAccum = 0
	d.filterSecondaryPeak = 0
	d.filterFloorLowLimit = 0
	d.filterFloorHighLimit = 0
	d.consistentEstimateCounter = 0
	d.consistentDelayReference = -10
}

func (d *stConsistentFilterDetector) Detect(filterToAnalyze []float32, region FilterRegion, xBlock *Block,
	peakIndex, delayBlocks int) bool {
	n := len(filterToAnalyze)
	if region.StartSample == 0 {
		d.filterFloorAccum = 0
		d.filterSecondaryPeak = 0
		if peakIndex < 64 {
			d.filterFloorLowLimit = 0
		} else {
			d.filterFloorLowLimit = peakIndex - 64
		}
		// The C++ compares against an unsigned size()-129.
		if uint64(peakIndex) > uint64(n)-129 {
			d.filterFloorHighLimit = 0
		} else {
			d.filterFloorHighLimit = peakIndex + 128
		}
	}
	filterFloorAccum := d.filterFloorAccum
	filterSecondaryPeak := d.filterSecondaryPeak
	for k := region.StartSample; k < min(region.EndSample+1, d.filterFloorLowLimit); k++ {
		absH := fabsf(filterToAnalyze[k])
		filterFloorAccum += absH
		filterSecondaryPeak = stdMax(filterSecondaryPeak, absH)
	}
	for k := max(d.filterFloorHighLimit, region.StartSample); k <= region.EndSample; k++ {
		absH := fabsf(filterToAnalyze[k])
		filterFloorAccum += absH
		filterSecondaryPeak = stdMax(filterSecondaryPeak, absH)
	}
	d.filterFloorAccum = filterFloorAccum
	d.filterSecondaryPeak = filterSecondaryPeak

	if region.EndSample == n-1 {
		filterFloor := d.filterFloorAccum / float32(uint64(d.filterFloorLowLimit+n-d.filterFloorHighLimit))
		absPeak := fabsf(filterToAnalyze[peakIndex])
		d.significantPeak = absPeak > 10*filterFloor && absPeak > 2*d.filterSecondaryPeak
	}

	if d.significantPeak {
		activeRenderBlock := false
		for ch := 0; ch < xBlock.NumChannels(); ch++ {
			xChannel := xBlock.View(0, ch)
			var xEnergy float32
			for _, v := range xChannel {
				xEnergy += float32(v * v)
			}
			if xEnergy > d.activeRenderThreshold {
				activeRenderBlock = true
				break
			}
		}
		if d.consistentDelayReference == delayBlocks {
			if activeRenderBlock {
				d.consistentEstimateCounter++
			}
		} else {
			d.consistentEstimateCounter = 0
			d.consistentDelayReference = delayBlocks
		}
	}
	return float32(d.consistentEstimateCounter) > 1.5*NumBlocksPerSecond
}

type stFilterAnalysisState struct {
	gain                     float32
	peakIndex                int
	filterLengthBlocks       int
	consistentEstimate       bool
	consistentFilterDetector stConsistentFilterDetector
}

func (s *stFilterAnalysisState) Reset(defaultGain float32) {
	s.peakIndex = 0
	s.gain = defaultGain
	s.consistentFilterDetector.Reset()
}

// FilterAnalyzer mirrors filter_analyzer.h: analyzes the adaptive filter
// impulse responses for delay, gain and consistency.
type FilterAnalyzer struct {
	boundedErl           bool
	defaultGain          float32
	hHighpass            [][]float32
	blocksSinceReset     int
	region               FilterRegion
	filterAnalysisStates []stFilterAnalysisState
	filterDelaysBlocks   []int
	minFilterDelayBlocks int

	// The per-channel state above for up to inlineChannels channels; the
	// high-pass filtered impulse responses of all channels share one array.
	inline struct {
		hHighpass            [inlineChannels][]float32
		filterAnalysisStates [inlineChannels]stFilterAnalysisState
		filterDelaysBlocks   [inlineChannels]int
	}
}

func NewFilterAnalyzer(config *EchoCanceller3Config, numCaptureChannels int) *FilterAnalyzer {
	f := &FilterAnalyzer{}
	f.init(config, numCaptureChannels)
	return f
}

// init sets up f in place; f must not be copied afterwards (see
// channelSlice).
func (f *FilterAnalyzer) init(config *EchoCanceller3Config, numCaptureChannels int) {
	*f = FilterAnalyzer{
		boundedErl:  config.EpStrength.BoundedErl,
		defaultGain: config.EpStrength.DefaultGain,
	}
	f.hHighpass = channelSlice(f.inline.hHighpass[:], numCaptureChannels)
	f.filterAnalysisStates = channelSlice(f.inline.filterAnalysisStates[:], numCaptureChannels)
	f.filterDelaysBlocks = channelSlice(f.inline.filterDelaysBlocks[:], numCaptureChannels)
	activeRenderLimit := config.RenderLevels.ActiveRenderLimit
	n := GetTimeDomainLength(config.Filter.Refined.LengthBlocks)
	hHighpass := make([]float32, numCaptureChannels*n)
	for ch := 0; ch < numCaptureChannels; ch++ {
		f.hHighpass[ch] = hHighpass[ch*n : (ch+1)*n : (ch+1)*n]
		st := &f.filterAnalysisStates[ch]
		st.filterLengthBlocks = config.Filter.RefinedInitial.LengthBlocks
		st.consistentFilterDetector.activeRenderThreshold = activeRenderLimit * activeRenderLimit * FftLengthBy2
		st.consistentFilterDetector.Reset()
		st.Reset(config.EpStrength.DefaultGain)
	}
	f.Reset()
}

func (f *FilterAnalyzer) Reset() {
	f.blocksSinceReset = 0
	f.resetRegion()
	for ch := range f.filterAnalysisStates {
		f.filterAnalysisStates[ch].Reset(f.defaultGain)
	}
	clear(f.filterDelaysBlocks)
}

// Update analyzes the filters and returns whether any filter is consistent
// and the maximum echo path gain.
func (f *FilterAnalyzer) Update(filtersTimeDomain [][]float32, renderBuffer *RenderBuffer) (anyFilterConsistent bool, maxEchoPathGain float32) {
	f.blocksSinceReset++
	f.SetRegionToAnalyze(len(filtersTimeDomain[0]))
	f.analyzeRegion(filtersTimeDomain, renderBuffer)
	stCh0 := &f.filterAnalysisStates[0]
	anyFilterConsistent = stCh0.consistentEstimate
	maxEchoPathGain = stCh0.gain
	f.minFilterDelayBlocks = f.filterDelaysBlocks[0]
	for ch := 1; ch < len(filtersTimeDomain); ch++ {
		stCh := &f.filterAnalysisStates[ch]
		anyFilterConsistent = anyFilterConsistent || stCh.consistentEstimate
		maxEchoPathGain = stdMax(maxEchoPathGain, stCh.gain)
		f.minFilterDelayBlocks = min(f.minFilterDelayBlocks, f.filterDelaysBlocks[ch])
	}
	return
}

// FilterDelaysBlocks returns the per-channel filter delays.
func (f *FilterAnalyzer) FilterDelaysBlocks() []int { return f.filterDelaysBlocks }

// MinFilterDelayBlocks returns the minimum delay over channels.
func (f *FilterAnalyzer) MinFilterDelayBlocks() int { return f.minFilterDelayBlocks }

// FilterLengthBlocks returns the filter length of channel 0.
func (f *FilterAnalyzer) FilterLengthBlocks() int {
	return f.filterAnalysisStates[0].filterLengthBlocks
}

// GetAdjustedFilters returns the high-pass filtered impulse responses.
func (f *FilterAnalyzer) GetAdjustedFilters() [][]float32 { return f.hHighpass }

func (f *FilterAnalyzer) analyzeRegion(filtersTimeDomain [][]float32, renderBuffer *RenderBuffer) {
	f.preProcessFilters(filtersTimeDomain)
	const oneByBlockSize float32 = 1.0 / BlockSize
	for ch := range filtersTimeDomain {
		stCh := &f.filterAnalysisStates[ch]
		stCh.peakIndex = min(stCh.peakIndex, len(f.hHighpass[ch])-1)
		stCh.peakIndex = stFindPeakIndex(f.hHighpass[ch], stCh.peakIndex, f.region.StartSample, f.region.EndSample)
		f.filterDelaysBlocks[ch] = stCh.peakIndex >> BlockSizeLog2
		f.updateFilterGain(f.hHighpass[ch], stCh)
		stCh.filterLengthBlocks = int(float32(len(filtersTimeDomain[ch])) * oneByBlockSize)
		xBlock := renderBuffer.GetBlock(-f.filterDelaysBlocks[ch])
		stCh.consistentEstimate = stCh.consistentFilterDetector.Detect(f.hHighpass[ch], f.region,
			&xBlock, stCh.peakIndex, f.filterDelaysBlocks[ch])
	}
}

func (f *FilterAnalyzer) updateFilterGain(filterTimeDomain []float32, st *stFilterAnalysisState) {
	sufficientTimeToConverge := f.blocksSinceReset > 5*NumBlocksPerSecond
	if sufficientTimeToConverge && st.consistentEstimate {
		st.gain = fabsf(filterTimeDomain[st.peakIndex])
	} else {
		if st.gain != 0 {
			st.gain = stdMax(st.gain, fabsf(filterTimeDomain[st.peakIndex]))
		}
	}
	if f.boundedErl && st.gain != 0 {
		st.gain = stdMax(st.gain, 0.01)
	}
}

var stHighpassCoeffs = [3]float32{0.7929742, -0.36072128, -0.47047766}

func (f *FilterAnalyzer) preProcessFilters(filtersTimeDomain [][]float32) {
	for ch := range filtersTimeDomain {
		n := len(filtersTimeDomain[ch])
		old := len(f.hHighpass[ch])
		if n <= cap(f.hHighpass[ch]) {
			f.hHighpass[ch] = f.hHighpass[ch][:n]
		} else {
			grown := make([]float32, n)
			copy(grown, f.hHighpass[ch])
			f.hHighpass[ch] = grown
		}
		if n > old {
			clear(f.hHighpass[ch][old:])
		}
		h := &stHighpassCoeffs
		hp := f.hHighpass[ch]
		clear(hp[f.region.StartSample : f.region.EndSample+1])
		x := filtersTimeDomain[ch]
		// hp[k] was cleared above, so each output is +0 plus the three
		// products in tap order.
		kernFir3(x, h, hp, max(len(h)-1, f.region.StartSample), f.region.EndSample)
	}
}

func (f *FilterAnalyzer) resetRegion() {
	f.region.StartSample = 0
	f.region.EndSample = 0
}

// SetRegionToAnalyze advances the analyzed region by one block.
func (f *FilterAnalyzer) SetRegionToAnalyze(filterSize int) {
	const numberBlocksToUpdate = 1
	r := &f.region
	if r.EndSample >= filterSize-1 {
		r.StartSample = 0
	} else {
		r.StartSample = r.EndSample + 1
	}
	r.EndSample = min(r.StartSample+numberBlocksToUpdate*BlockSize-1, filterSize-1)
}
