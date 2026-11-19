package aec3

// ReverbModelEstimator mirrors reverb_model_estimator.h: estimates the model
// parameters of the reverberant echo.
type ReverbModelEstimator struct {
	reverbDecayEstimators    []ReverbDecayEstimator
	reverbFrequencyResponses []ReverbFrequencyResponse

	// The per-channel estimators for up to inlineChannels channels; the
	// samples of the decay estimators of all channels share one array.
	inline struct {
		reverbDecayEstimators    [inlineChannels]ReverbDecayEstimator
		reverbFrequencyResponses [inlineChannels]ReverbFrequencyResponse
	}
}

func NewReverbModelEstimator(config *EchoCanceller3Config, numCaptureChannels int) *ReverbModelEstimator {
	r := &ReverbModelEstimator{}
	r.init(config, numCaptureChannels)
	return r
}

// init sets up r in place; r must not be copied afterwards (see
// channelSlice).
func (r *ReverbModelEstimator) init(config *EchoCanceller3Config, numCaptureChannels int) {
	*r = ReverbModelEstimator{}
	r.reverbDecayEstimators = channelSlice(r.inline.reverbDecayEstimators[:], numCaptureChannels)
	r.reverbFrequencyResponses = channelSlice(r.inline.reverbFrequencyResponses[:], numCaptureChannels)
	n := stReverbDecayStorageSize(config)
	storage := make([]float32, numCaptureChannels*n)
	for ch := range r.reverbDecayEstimators {
		r.reverbDecayEstimators[ch].init(config, storage[ch*n:(ch+1)*n:(ch+1)*n])
		r.reverbFrequencyResponses[ch] = ReverbFrequencyResponse{
			useConservativeTailFrequencyResponse: config.EpStrength.UseConservativeTailFrequencyResponse}
	}
}

// Update updates the estimates based on new data.
func (r *ReverbModelEstimator) Update(impulseResponses [][]float32, frequencyResponses [][][FftLengthBy2Plus1]float32,
	linearFilterQualities []LinearFilterQuality, filterDelaysBlocks []int, usableLinearEstimates []bool,
	stationaryBlock bool) {
	for ch := range r.reverbDecayEstimators {
		r.reverbFrequencyResponses[ch].Update(frequencyResponses[ch], filterDelaysBlocks[ch],
			linearFilterQualities[ch], stationaryBlock)
		r.reverbDecayEstimators[ch].Update(impulseResponses[ch], linearFilterQualities[ch],
			filterDelaysBlocks[ch], usableLinearEstimates[ch], stationaryBlock)
	}
}

// ReverbDecay returns the exponential decay of the reverberant echo (channel
// 0); mild selects the milder decay.
func (r *ReverbModelEstimator) ReverbDecay(mild bool) float32 {
	return r.reverbDecayEstimators[0].Decay(mild)
}

// GetReverbFrequencyResponse returns the frequency response of the
// reverberant echo (channel 0).
func (r *ReverbModelEstimator) GetReverbFrequencyResponse() *[FftLengthBy2Plus1]float32 {
	return r.reverbFrequencyResponses[0].FrequencyResponse()
}
