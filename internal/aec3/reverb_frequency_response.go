package aec3

// ReverbFrequencyResponse mirrors reverb_frequency_response.h: the frequency
// response estimate of the reverb tail.
type ReverbFrequencyResponse struct {
	useConservativeTailFrequencyResponse bool
	averageDecay                         float32
	tailResponse                         [FftLengthBy2Plus1]float32
}

func NewReverbFrequencyResponse(useConservativeTailFrequencyResponse bool) *ReverbFrequencyResponse {
	return &ReverbFrequencyResponse{useConservativeTailFrequencyResponse: useConservativeTailFrequencyResponse}
}

// FrequencyResponse returns the estimated frequency response of the reverb.
func (r *ReverbFrequencyResponse) FrequencyResponse() *[FftLengthBy2Plus1]float32 {
	return &r.tailResponse
}

// stAverageDecayWithinFilter is the tail/direct-path energy ratio, skipping
// the DC bin.
func stAverageDecayWithinFilter(directPath, tail *[FftLengthBy2Plus1]float32) float32 {
	directPathEnergy := stAccumulate(directPath[1:])
	if directPathEnergy == 0 {
		return 0
	}
	tailEnergy := stAccumulate(tail[1:])
	return tailEnergy / directPathEnergy
}

// Update updates the frequency response estimate of the reverb.
func (r *ReverbFrequencyResponse) Update(frequencyResponse [][FftLengthBy2Plus1]float32, filterDelayBlocks int,
	linearFilterQuality LinearFilterQuality, stationaryBlock bool) {
	if stationaryBlock || !linearFilterQuality.Ok {
		return
	}
	r.update(frequencyResponse, filterDelayBlocks, linearFilterQuality.Value)
}

func (r *ReverbFrequencyResponse) update(frequencyResponse [][FftLengthBy2Plus1]float32, filterDelayBlocks int,
	linearFilterQuality float32) {
	tail := &frequencyResponse[len(frequencyResponse)-1]
	directPath := &frequencyResponse[filterDelayBlocks]
	averageDecay := stAverageDecayWithinFilter(directPath, tail)
	smoothing := 0.2 * linearFilterQuality
	r.averageDecay += float32(smoothing * (averageDecay - r.averageDecay))
	for k := 0; k < FftLengthBy2Plus1; k++ {
		r.tailResponse[k] = directPath[k] * r.averageDecay
	}
	if r.useConservativeTailFrequencyResponse {
		for k := 0; k < FftLengthBy2Plus1; k++ {
			r.tailResponse[k] = stMaxf(tail[k], r.tailResponse[k])
		}
	}
	for k := 1; k < FftLengthBy2; k++ {
		avgNeighbour := 0.5 * (r.tailResponse[k-1] + r.tailResponse[k+1])
		r.tailResponse[k] = stMaxf(r.tailResponse[k], avgNeighbour)
	}
}
