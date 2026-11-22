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

// FilterRegion is the part of the filter analyzed in one block.
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
		// The C++ compares with size()-129, which is unsigned.
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

// FilterAnalyzer analyzes the impulse responses of the adaptive filters for
// delay, gain and consistency (filter_analyzer.h).
//
// The high-pass filtered impulse responses share storage, hCap samples per
// channel: room for the longest filter of the config.
type FilterAnalyzer struct {
	*filterAnalyzerState
	more    *faChannels // with more than inlineChannels channels
	storage []float32
}

// filterAnalyzerState is the state of a FilterAnalyzer except its pointers.
type filterAnalyzerState struct {
	storageSpan          span
	numCaptureChannels   int
	hCap                 int
	boundedErl           bool
	defaultGain          float32
	blocksSinceReset     int
	region               FilterRegion
	minFilterDelayBlocks int

	inline struct {
		hLen                 [inlineChannels]int
		filterAnalysisStates [inlineChannels]stFilterAnalysisState
		filterDelaysBlocks   [inlineChannels]int
	}
}

// faChannels is the per-channel state of a FilterAnalyzer: the length of the
// high-pass filtered impulse response, the analysis state and the delay.
type faChannels struct {
	hLen                 []int
	filterAnalysisStates []stFilterAnalysisState
	filterDelaysBlocks   []int
}

func (f FilterAnalyzer) channels() faChannels {
	if f.more != nil {
		return *f.more
	}
	n := f.numCaptureChannels
	return faChannels{
		hLen:                 f.inline.hLen[:n:n],
		filterAnalysisStates: f.inline.filterAnalysisStates[:n:n],
		filterDelaysBlocks:   f.inline.filterDelaysBlocks[:n:n],
	}
}

// hHighpass returns the n samples of the high-pass filtered impulse response
// of channel ch.
func (f FilterAnalyzer) hHighpass(ch, n int) []float32 {
	return f.storage[ch*f.hCap : ch*f.hCap+n : (ch+1)*f.hCap]
}

func (f *FilterAnalyzer) bind(st *filterAnalyzerState, b *binder) {
	f.filterAnalyzerState = st
	b.f32(&f.storage, &st.storageSpan)
	if b != nil {
		bindPtr(b, &f.more, &b.x.faMore)
	}
}

func NewFilterAnalyzer(config *EchoCanceller3Config, numCaptureChannels int) *FilterAnalyzer {
	f := &FilterAnalyzer{}
	f.bind(&filterAnalyzerState{}, bindAlone)
	f.init(config, numCaptureChannels, make([]float32, faStorageSize(config, numCaptureChannels)))
	return f
}

// faStorageSize returns the number of samples a FilterAnalyzer keeps for
// config.
func faStorageSize(config *EchoCanceller3Config, numCaptureChannels int) int {
	return numCaptureChannels * faChannelSize(config)
}

// faChannelSize returns the number of samples a FilterAnalyzer keeps per
// channel for config: those of the longest refined filter, the initial one
// included.
func faChannelSize(config *EchoCanceller3Config) int {
	return GetTimeDomainLength(max(config.Filter.Refined.LengthBlocks, config.Filter.RefinedInitial.LengthBlocks))
}

// init sets up f in place. storage holds faStorageSize zeros.
func (f *FilterAnalyzer) init(config *EchoCanceller3Config, numCaptureChannels int, storage []float32) {
	st := f.filterAnalyzerState
	*st = filterAnalyzerState{
		numCaptureChannels: numCaptureChannels,
		hCap:               faChannelSize(config),
		boundedErl:         config.EpStrength.BoundedErl,
		defaultGain:        config.EpStrength.DefaultGain,
	}
	*f = FilterAnalyzer{storage: storage}
	f.bind(st, bindAlone)
	if n := numCaptureChannels; n > inlineChannelLimit {
		f.more = &faChannels{
			hLen:                 make([]int, n),
			filterAnalysisStates: make([]stFilterAnalysisState, n),
			filterDelaysBlocks:   make([]int, n),
		}
	}
	activeRenderLimit := config.RenderLevels.ActiveRenderLimit
	c := f.channels()
	for ch := 0; ch < numCaptureChannels; ch++ {
		c.hLen[ch] = f.hCap
		st := &c.filterAnalysisStates[ch]
		st.filterLengthBlocks = config.Filter.RefinedInitial.LengthBlocks
		st.consistentFilterDetector.activeRenderThreshold = activeRenderLimit * activeRenderLimit * FftLengthBy2
		st.consistentFilterDetector.Reset()
		st.Reset(config.EpStrength.DefaultGain)
	}
	f.Reset()
}

func (f FilterAnalyzer) Reset() {
	f.blocksSinceReset = 0
	f.resetRegion()
	c := f.channels()
	for ch := range c.filterAnalysisStates {
		c.filterAnalysisStates[ch].Reset(f.defaultGain)
	}
	clear(c.filterDelaysBlocks)
}

// Update analyzes the filters. It reports whether any filter is consistent
// and returns the largest echo path gain.
func (f *FilterAnalyzer) Update(filtersTimeDomain [][]float32, renderBuffer *RenderBuffer) (anyFilterConsistent bool, maxEchoPathGain float32) {
	f.blocksSinceReset++
	f.SetRegionToAnalyze(len(filtersTimeDomain[0]))
	c := f.channels()
	f.analyzeRegion(c, filtersTimeDomain, renderBuffer)
	stCh0 := &c.filterAnalysisStates[0]
	anyFilterConsistent = stCh0.consistentEstimate
	maxEchoPathGain = stCh0.gain
	f.minFilterDelayBlocks = c.filterDelaysBlocks[0]
	for ch := 1; ch < len(filtersTimeDomain); ch++ {
		stCh := &c.filterAnalysisStates[ch]
		anyFilterConsistent = anyFilterConsistent || stCh.consistentEstimate
		maxEchoPathGain = stdMax(maxEchoPathGain, stCh.gain)
		f.minFilterDelayBlocks = min(f.minFilterDelayBlocks, c.filterDelaysBlocks[ch])
	}
	return
}

// FilterDelaysBlocks returns the filter delays, one per channel.
func (f FilterAnalyzer) FilterDelaysBlocks() []int { return f.channels().filterDelaysBlocks }

// MinFilterDelayBlocks returns the smallest filter delay of the channels.
func (f FilterAnalyzer) MinFilterDelayBlocks() int { return f.minFilterDelayBlocks }

// FilterLengthBlocks returns the filter length of channel 0.
func (f FilterAnalyzer) FilterLengthBlocks() int {
	return f.channels().filterAnalysisStates[0].filterLengthBlocks
}

// GetAdjustedFilters returns the high-pass filtered impulse responses.
func (f FilterAnalyzer) GetAdjustedFilters() [][]float32 {
	return f.adjustedFilters(make([][]float32, f.numCaptureChannels))
}

// adjustedFilters is GetAdjustedFilters into dst, which has one element per
// channel.
func (f FilterAnalyzer) adjustedFilters(dst [][]float32) [][]float32 {
	c := f.channels()
	dst = dst[:f.numCaptureChannels]
	for ch := range dst {
		dst[ch] = f.hHighpass(ch, c.hLen[ch])
	}
	return dst
}

func (f *FilterAnalyzer) analyzeRegion(c faChannels, filtersTimeDomain [][]float32, renderBuffer *RenderBuffer) {
	f.preProcessFilters(c, filtersTimeDomain)
	const oneByBlockSize float32 = 1.0 / BlockSize
	for ch := range filtersTimeDomain {
		stCh := &c.filterAnalysisStates[ch]
		hp := f.hHighpass(ch, c.hLen[ch])
		stCh.peakIndex = min(stCh.peakIndex, len(hp)-1)
		stCh.peakIndex = stFindPeakIndex(hp, stCh.peakIndex, f.region.StartSample, f.region.EndSample)
		c.filterDelaysBlocks[ch] = stCh.peakIndex >> BlockSizeLog2
		f.updateFilterGain(hp, stCh)
		stCh.filterLengthBlocks = int(float32(len(filtersTimeDomain[ch])) * oneByBlockSize)
		xBlock := renderBuffer.GetBlock(-c.filterDelaysBlocks[ch])
		stCh.consistentEstimate = stCh.consistentFilterDetector.Detect(hp, f.region,
			&xBlock, stCh.peakIndex, c.filterDelaysBlocks[ch])
	}
}

func (f FilterAnalyzer) updateFilterGain(filterTimeDomain []float32, st *stFilterAnalysisState) {
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

// grow gives every channel room for n samples. Only a FilterAnalyzer that
// stands alone grows.
func (f *FilterAnalyzer) grow(c faChannels, n int) {
	storage := make([]float32, f.numCaptureChannels*n)
	for ch := range c.hLen {
		copy(storage[ch*n:(ch+1)*n], f.hHighpass(ch, c.hLen[ch]))
	}
	f.storage, f.hCap = storage, n
}

func (f *FilterAnalyzer) preProcessFilters(c faChannels, filtersTimeDomain [][]float32) {
	for ch := range filtersTimeDomain {
		n := len(filtersTimeDomain[ch])
		old := c.hLen[ch]
		if n > f.hCap {
			f.grow(c, n)
		}
		c.hLen[ch] = n
		hp := f.hHighpass(ch, n)
		if n > old {
			clear(hp[old:])
		}
		h := &stHighpassCoeffs
		clear(hp[f.region.StartSample : f.region.EndSample+1])
		x := filtersTimeDomain[ch]
		// hp[k] is +0 here, so an output is +0 plus the three products in
		// tap order.
		kernFir3(x, h, hp, max(len(h)-1, f.region.StartSample), f.region.EndSample)
	}
}

func (f FilterAnalyzer) resetRegion() {
	f.region.StartSample = 0
	f.region.EndSample = 0
}

// SetRegionToAnalyze advances the analyzed region by one block.
func (f FilterAnalyzer) SetRegionToAnalyze(filterSize int) {
	const numberBlocksToUpdate = 1
	r := &f.region
	if r.EndSample >= filterSize-1 {
		r.StartSample = 0
	} else {
		r.StartSample = r.EndSample + 1
	}
	r.EndSample = min(r.StartSample+numberBlocksToUpdate*BlockSize-1, filterSize-1)
}
