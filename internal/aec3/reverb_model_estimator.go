package aec3

// ReverbModelEstimator mirrors reverb_model_estimator.h: estimates the model
// parameters of the reverberant echo.
//
// The samples of the decay estimators of all channels share one array
// (storage), perChannel values each.
type ReverbModelEstimator struct {
	*reverbModelEstimatorState
	more    *rmChannels // with more than inlineChannels channels
	storage []float32
}

// reverbModelEstimatorState is the state of a ReverbModelEstimator but its
// pointers.
type reverbModelEstimatorState struct {
	storageSpan        span
	numCaptureChannels int
	perChannel         int

	// The per-channel estimators for up to inlineChannels channels.
	inline struct {
		reverbDecayEstimators    [inlineChannels]reverbDecayState
		reverbFrequencyResponses [inlineChannels]ReverbFrequencyResponse
	}
}

// rmChannels is the per-channel estimators of a ReverbModelEstimator.
type rmChannels struct {
	reverbDecayEstimators    []reverbDecayState
	reverbFrequencyResponses []ReverbFrequencyResponse
}

// channels returns the per-channel estimators: views of r for up to
// inlineChannels channels.
func (r ReverbModelEstimator) channels() rmChannels {
	if r.more != nil {
		return *r.more
	}
	n := r.numCaptureChannels
	return rmChannels{
		reverbDecayEstimators:    r.inline.reverbDecayEstimators[:n:n],
		reverbFrequencyResponses: r.inline.reverbFrequencyResponses[:n:n],
	}
}

// bind makes r the view of st.
func (r *ReverbModelEstimator) bind(st *reverbModelEstimatorState, b *binder) {
	r.reverbModelEstimatorState = st
	b.f32(&r.storage, &st.storageSpan)
	if b != nil {
		bindPtr(b, &r.more, &b.x.rmMore)
	}
}

func NewReverbModelEstimator(config *EchoCanceller3Config, numCaptureChannels int) *ReverbModelEstimator {
	r := &ReverbModelEstimator{}
	r.bind(&reverbModelEstimatorState{}, bindAlone)
	r.init(config, numCaptureChannels, make([]float32, rmStorageSize(config, numCaptureChannels)))
	return r
}

// rmStorageSize returns the number of samples a ReverbModelEstimator keeps
// for config.
func rmStorageSize(config *EchoCanceller3Config, numCaptureChannels int) int {
	return numCaptureChannels * stReverbDecayStorageSize(config)
}

// init sets up r in place over storage, which holds rmStorageSize zeros.
func (r *ReverbModelEstimator) init(config *EchoCanceller3Config, numCaptureChannels int, storage []float32) {
	st := r.reverbModelEstimatorState
	*st = reverbModelEstimatorState{
		numCaptureChannels: numCaptureChannels,
		perChannel:         stReverbDecayStorageSize(config),
	}
	*r = ReverbModelEstimator{storage: storage}
	r.bind(st, bindAlone)
	if n := numCaptureChannels; n > inlineChannelLimit {
		r.more = &rmChannels{
			reverbDecayEstimators:    make([]reverbDecayState, n),
			reverbFrequencyResponses: make([]ReverbFrequencyResponse, n),
		}
	}
	c := r.channels()
	for ch := range c.reverbDecayEstimators {
		c.reverbDecayEstimators[ch].init(config)
		c.reverbFrequencyResponses[ch] = ReverbFrequencyResponse{
			useConservativeTailFrequencyResponse: config.EpStrength.UseConservativeTailFrequencyResponse}
	}
}

// Update updates the estimates based on new data.
func (r ReverbModelEstimator) Update(impulseResponses [][]float32, frequencyResponses [][][FftLengthBy2Plus1]float32,
	linearFilterQualities []LinearFilterQuality, filterDelaysBlocks []int, usableLinearEstimates []bool,
	stationaryBlock bool) {
	c := r.channels()
	n := r.perChannel
	for ch := range c.reverbDecayEstimators {
		c.reverbFrequencyResponses[ch].Update(frequencyResponses[ch], filterDelaysBlocks[ch],
			linearFilterQualities[ch], stationaryBlock)
		c.reverbDecayEstimators[ch].Update(r.storage[ch*n:(ch+1)*n:(ch+1)*n], impulseResponses[ch], linearFilterQualities[ch],
			filterDelaysBlocks[ch], usableLinearEstimates[ch], stationaryBlock)
	}
}

// ReverbDecay returns the exponential decay of the reverberant echo (channel
// 0); mild selects the milder decay.
func (r ReverbModelEstimator) ReverbDecay(mild bool) float32 {
	return r.channels().reverbDecayEstimators[0].Decay(mild)
}

// GetReverbFrequencyResponse returns the frequency response of the
// reverberant echo (channel 0).
func (r ReverbModelEstimator) GetReverbFrequencyResponse() *[FftLengthBy2Plus1]float32 {
	return r.channels().reverbFrequencyResponses[0].FrequencyResponse()
}
