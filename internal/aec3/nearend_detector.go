package aec3

// NearendDetector selects the nearend state from spectral estimates.
type NearendDetector interface {
	// IsNearendState returns whether the current state is the nearend state.
	IsNearendState() bool
	// Update updates the state selection based on latest spectral estimates.
	Update(nearendSpectrum, residualEchoSpectrum, comfortNoiseSpectrum [][FftLengthBy2Plus1]float32,
		initialState bool)
}
