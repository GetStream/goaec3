package aec3

// ReverbFrequencyResponse estimates the frequency response of the reverb
// tail (reverb_frequency_response.h).
type ReverbFrequencyResponse struct {
	useConservativeTailFrequencyResponse bool
	averageDecay                         float32
	tailResponse                         [FftLengthBy2Plus1]float32
}

func NewReverbFrequencyResponse(useConservativeTailFrequencyResponse bool) *ReverbFrequencyResponse {
	return &ReverbFrequencyResponse{useConservativeTailFrequencyResponse: useConservativeTailFrequencyResponse}
}

// FrequencyResponse returns the estimate.
func (r *ReverbFrequencyResponse) FrequencyResponse() *[FftLengthBy2Plus1]float32 {
	return &r.tailResponse
}

// stAverageDecayWithinFilter returns the ratio of the tail energy to the
// direct path energy, without the first bin. Both energies accumulate in bin
// order; they are independent, so the chains run side by side.
func stAverageDecayWithinFilter(directPath, tail *[FftLengthBy2Plus1]float32) float32 {
	var directPathEnergy, tailEnergy float32
	for k := 1; k < FftLengthBy2Plus1; k++ {
		directPathEnergy += directPath[k]
		tailEnergy += tail[k]
	}
	if directPathEnergy == 0 {
		return 0
	}
	return tailEnergy / directPathEnergy
}

// Update takes the frequency response of the filter of one block.
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
	kernScale(directPath[:], r.averageDecay, r.tailResponse[:])
	if r.useConservativeTailFrequencyResponse {
		kernStdMax(tail[:], r.tailResponse[:], r.tailResponse[:])
	}
	// A bin depends on the updated bin before it, which stays in a register.
	t := &r.tailResponse
	prev := t[0]
	for k := 1; k < FftLengthBy2; k++ {
		avgNeighbour := 0.5 * (prev + t[k+1])
		prev = stdMax(t[k], avgNeighbour)
		t[k] = prev
	}
}
