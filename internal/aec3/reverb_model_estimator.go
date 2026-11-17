package aec3

// ReverbModelEstimator mirrors reverb_model_estimator.h: estimates the model
// parameters of the reverberant echo.
type ReverbModelEstimator struct {
	reverbDecayEstimators    []*ReverbDecayEstimator
	reverbFrequencyResponses []*ReverbFrequencyResponse
}

func NewReverbModelEstimator(config *EchoCanceller3Config, numCaptureChannels int) *ReverbModelEstimator {
	r := &ReverbModelEstimator{
		reverbDecayEstimators:    make([]*ReverbDecayEstimator, numCaptureChannels),
		reverbFrequencyResponses: make([]*ReverbFrequencyResponse, numCaptureChannels),
	}
	for ch := range r.reverbDecayEstimators {
		r.reverbDecayEstimators[ch] = NewReverbDecayEstimator(config)
		r.reverbFrequencyResponses[ch] = NewReverbFrequencyResponse(config.EpStrength.UseConservativeTailFrequencyResponse)
	}
	return r
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
