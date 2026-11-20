package aec3

import "math"

// SuppressionFilter applies the suppression gains and comfort noise in the
// frequency domain and synthesizes the time-domain output of every band.
type SuppressionFilter struct {
	*suppressionFilterState
	more *sfMore // with more than inlineChannels channels
}

// suppressionFilterState is the state of a SuppressionFilter but its
// pointer.
type suppressionFilterState struct {
	optimization       Aec3Optimization
	sampleRateHz       int
	numCaptureChannels int
	fft                Aec3Fft
	fftPair            aec3FftPair
	eOutputOld         [inlineChannels][MaxNumBands][FftLengthBy2]float32 // [channel][band]

	// Scratch.
	e         FftData
	eExtended [FftLength]float32
}

// sfMore holds the previous output blocks of a SuppressionFilter with more
// than inlineChannels capture channels.
type sfMore struct {
	eOutputOld [][MaxNumBands][FftLengthBy2]float32 // [channel][band]
}

// bind makes f the view of st.
func (f *SuppressionFilter) bind(st *suppressionFilterState, b *binder) {
	f.suppressionFilterState = st
	if b != nil {
		bindPtr(b, &f.more, &b.x.sfMore)
	}
}

func NewSuppressionFilter(optimization Aec3Optimization, sampleRateHz, numCaptureChannels int) *SuppressionFilter {
	f := &SuppressionFilter{}
	f.bind(&suppressionFilterState{}, nil)
	f.init(optimization, sampleRateHz, numCaptureChannels)
	return f
}

// init sets up f in place.
func (f *SuppressionFilter) init(optimization Aec3Optimization, sampleRateHz, numCaptureChannels int) {
	st := f.suppressionFilterState
	*st = suppressionFilterState{}
	*f = SuppressionFilter{}
	f.bind(st, nil)
	f.optimization = optimization
	f.sampleRateHz = sampleRateHz
	f.numCaptureChannels = numCaptureChannels
	if numCaptureChannels > inlineChannelLimit {
		f.more = &sfMore{eOutputOld: make([][MaxNumBands][FftLengthBy2]float32, numCaptureChannels)}
	}
}

// ApplyGain applies the suppression gain, adds the comfort noise and writes
// the synthesized output into e.
func (f *SuppressionFilter) ApplyGain(comfortNoise, comfortNoiseHighBand []FftData,
	suppressionGain *[FftLengthBy2Plus1]float32, highBandsGain float32,
	eLowestBand []FftData, e *Block) {
	// Comfort noise gain is sqrt(1-g^2), where g is the suppression gain.
	var noiseGain [FftLengthBy2Plus1]float32
	kernOneMinusSq(suppressionGain[:], noiseGain[:])
	NewVectorMath(f.optimization).Sqrt(noiseGain[:])

	highBandsNoiseScaling := 0.4 * float32(math.Sqrt(float64(1-float32(highBandsGain*highBandsGain))))

	numBands := e.NumBands()
	E := &f.e
	eExtended := &f.eExtended
	var more [][MaxNumBands][FftLengthBy2]float32
	if f.more != nil {
		more = f.more.eOutputOld
	}
	eOutputOld := channelState(&f.eOutputOld, f.numCaptureChannels, more)
	for ch := range eOutputOld {
		eOld := &eOutputOld[ch]
		// Analysis filterbank.
		E.Assign(&eLowestBand[ch])

		cn := &comfortNoise[ch]
		// Apply suppression gains, then scale and add the comfort noise:
		// E = float32(E*g) + float32(noiseGain*cn).
		kernMulAdd2(E.Re[:], suppressionGain[:], noiseGain[:], cn.Re[:], E.Re[:])
		kernMulAdd2(E.Im[:], suppressionGain[:], noiseGain[:], cn.Im[:], E.Im[:])

		// Synthesis filterbank.
		const ifftNormalization float32 = 2.0 / FftLength
		// With more than one band the comfort noise of band 1 is
		// synthesized too; the two inverse transforms run as a pair. The
		// transform reads neither Im[0] nor Im[FftLengthBy2], which is all
		// that FftData::Assign changes in the comfort noise.
		eNoise := &f.fftPair.x0
		if numBands > 1 {
			f.fftPair.Ifft(E, &comfortNoiseHighBand[ch], eExtended, eNoise)
		} else {
			f.fft.Ifft(E, eExtended)
		}

		e0 := e.View(0, ch)
		e0Old := &eOld[0]

		// Window and add the first half of e_extended with the second half of
		// e_extended from the previous block.
		e0 = e0[:FftLengthBy2]
		kernMulAdd2(e0Old[:], sqrtHanning128[FftLengthBy2:], eExtended[:FftLengthBy2], sqrtHanning128[:FftLengthBy2], e0)
		kernScale(e0, ifftNormalization, e0)

		// The second half of e_extended is stored for the succeeding frame.
		copy(e0Old[:], eExtended[FftLengthBy2:])

		// Apply suppression gain to upper bands.
		for b := 1; b < numBands; b++ {
			eBand := e.View(b, ch)
			kernScale(eBand, highBandsGain, eBand)
		}

		// Add comfort noise to band 1.
		if numBands > 1 {
			e1 := e.View(1, ch)
			gain := highBandsNoiseScaling * ifftNormalization
			kernAxpy(eNoise[:FftLengthBy2], gain, e1)
		}

		// Delay upper bands to match the delay of the filter bank.
		for b := 1; b < numBands; b++ {
			eBand := e.View(b, ch)
			eBandOld := &eOld[b]
			for i := 0; i < FftLengthBy2; i++ {
				eBand[i], eBandOld[i] = eBandOld[i], eBand[i]
			}
		}

		// Clamp output of all bands.
		for b := 0; b < numBands; b++ {
			// SafeClamp(x, -32768.f, 32767.f); see kernClamp.
			kernClamp(e.View(b, ch), -32768, 32767)
		}
	}
}
