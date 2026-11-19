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
// the DC bin. Both energies accumulate in bin order; the tail energy is only
// used with a nonzero direct path, and running the two chains side by side
// halves the wait for them.
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
	kernScale(directPath[:], r.averageDecay, r.tailResponse[:])
	if r.useConservativeTailFrequencyResponse {
		kernStdMax(tail[:], r.tailResponse[:], r.tailResponse[:])
	}
	// Each bin depends on the updated bin before it, which stays in a
	// register instead of being read back from the array.
	t := &r.tailResponse
	prev := t[0]
	for k := 1; k < FftLengthBy2; k++ {
		avgNeighbour := 0.5 * (prev + t[k+1])
		prev = stdMax(t[k], avgNeighbour)
		t[k] = prev
	}
}
