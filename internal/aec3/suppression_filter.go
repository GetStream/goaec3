package aec3

import "math"

// SuppressionFilter applies the suppression gains and comfort noise in the
// frequency domain and synthesizes the time-domain output of every band.
type SuppressionFilter struct {
	optimization       Aec3Optimization
	sampleRateHz       int
	numCaptureChannels int
	fft                Aec3Fft
	eOutputOld         [][][FftLengthBy2]float32 // [band][channel]

	// Scratch.
	e         FftData
	eExtended [FftLength]float32
}

func NewSuppressionFilter(optimization Aec3Optimization, sampleRateHz, numCaptureChannels int) *SuppressionFilter {
	f := &SuppressionFilter{
		optimization:       optimization,
		sampleRateHz:       sampleRateHz,
		numCaptureChannels: numCaptureChannels,
		eOutputOld:         make([][][FftLengthBy2]float32, NumBandsForRate(sampleRateHz)),
	}
	for b := range f.eOutputOld {
		f.eOutputOld[b] = make([][FftLengthBy2]float32, numCaptureChannels)
	}
	return f
}

// ApplyGain applies the suppression gain, adds the comfort noise and writes
// the synthesized output into e.
func (f *SuppressionFilter) ApplyGain(comfortNoise, comfortNoiseHighBand []FftData,
	suppressionGain *[FftLengthBy2Plus1]float32, highBandsGain float32,
	eLowestBand []FftData, e *Block) {
	// Comfort noise gain is sqrt(1-g^2), where g is the suppression gain.
	var noiseGain [FftLengthBy2Plus1]float32
	for i := range noiseGain {
		noiseGain[i] = 1 - float32(suppressionGain[i]*suppressionGain[i])
	}
	NewVectorMath(f.optimization).Sqrt(noiseGain[:])

	highBandsNoiseScaling := 0.4 * float32(math.Sqrt(float64(1-float32(highBandsGain*highBandsGain))))

	numBands := e.NumBands()
	E := &f.e
	eExtended := &f.eExtended
	for ch := 0; ch < f.numCaptureChannels; ch++ {
		// Analysis filterbank.
		E.Assign(&eLowestBand[ch])

		cn := &comfortNoise[ch]
		for i := 0; i < FftLengthBy2Plus1; i++ {
			// Apply suppression gains.
			eReal := float32(E.Re[i] * suppressionGain[i])
			eImag := float32(E.Im[i] * suppressionGain[i])

			// Scale and add the comfort noise.
			E.Re[i] = eReal + float32(noiseGain[i]*cn.Re[i])
			E.Im[i] = eImag + float32(noiseGain[i]*cn.Im[i])
		}

		// Synthesis filterbank.
		const ifftNormalization float32 = 2.0 / FftLength
		f.fft.Ifft(E, eExtended)

		e0 := e.View(0, ch)
		e0Old := &f.eOutputOld[0][ch]

		// Window and add the first half of e_extended with the second half of
		// e_extended from the previous block.
		for i := 0; i < FftLengthBy2; i++ {
			e0i := float32(e0Old[i] * sqrtHanning128[FftLengthBy2+i])
			e0i += float32(eExtended[i] * sqrtHanning128[i])
			e0[i] = e0i * ifftNormalization
		}

		// The second half of e_extended is stored for the succeeding frame.
		copy(e0Old[:], eExtended[FftLengthBy2:])

		// Apply suppression gain to upper bands.
		for b := 1; b < numBands; b++ {
			eBand := e.View(b, ch)
			for i := range eBand {
				eBand[i] *= highBandsGain
			}
		}

		// Add comfort noise to band 1.
		if numBands > 1 {
			E.Assign(&comfortNoiseHighBand[ch])
			f.fft.Ifft(E, eExtended)

			e1 := e.View(1, ch)
			gain := highBandsNoiseScaling * ifftNormalization
			for i := 0; i < FftLengthBy2; i++ {
				e1[i] += float32(eExtended[i] * gain)
			}
		}

		// Delay upper bands to match the delay of the filter bank.
		for b := 1; b < numBands; b++ {
			eBand := e.View(b, ch)
			eBandOld := &f.eOutputOld[b][ch]
			for i := 0; i < FftLengthBy2; i++ {
				eBand[i], eBandOld[i] = eBandOld[i], eBand[i]
			}
		}

		// Clamp output of all bands.
		for b := 0; b < numBands; b++ {
			eBand := e.View(b, ch)
			for i, v := range eBand {
				eBand[i] = supSafeClamp(v, -32768, 32767)
			}
		}
	}
}

func supSafeClamp(x, lo, hi float32) float32 {
	if x < lo {
		return lo
	}
	if hi < x {
		return hi
	}
	return x
}
