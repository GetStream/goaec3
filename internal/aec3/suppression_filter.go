package aec3

import "math"

// SuppressionFilter applies the suppression gains and the comfort noise in
// the frequency domain and synthesizes the output of every band
// (suppression_filter.h).
type SuppressionFilter struct {
	*suppressionFilterState
	more *sfMore // with more than inlineChannels channels
}

// suppressionFilterState is the state of a SuppressionFilter except its
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

func (f *SuppressionFilter) bind(st *suppressionFilterState, b *binder) {
	f.suppressionFilterState = st
	if b != nil {
		bindPtr(b, &f.more, &b.x.sfMore)
	}
}

func NewSuppressionFilter(optimization Aec3Optimization, sampleRateHz, numCaptureChannels int) *SuppressionFilter {
	f := &SuppressionFilter{}
	f.bind(&suppressionFilterState{}, bindAlone)
	f.init(optimization, sampleRateHz, numCaptureChannels)
	return f
}

func (f *SuppressionFilter) init(optimization Aec3Optimization, sampleRateHz, numCaptureChannels int) {
	st := f.suppressionFilterState
	*st = suppressionFilterState{}
	*f = SuppressionFilter{}
	f.bind(st, bindAlone)
	f.optimization = optimization
	f.sampleRateHz = sampleRateHz
	f.numCaptureChannels = numCaptureChannels
	if numCaptureChannels > inlineChannelLimit {
		f.more = &sfMore{eOutputOld: make([][MaxNumBands][FftLengthBy2]float32, numCaptureChannels)}
	}
}

// ApplyGain applies the suppression gains, adds the comfort noise and writes
// the synthesized output to e.
func (f SuppressionFilter) ApplyGain(comfortNoise, comfortNoiseHighBand []FftData,
	suppressionGain *[FftLengthBy2Plus1]float32, highBandsGain float32,
	eLowestBand []FftData, e *Block) {
	// The gain of the comfort noise is sqrt(1 - g*g) for a suppression gain
	// g.
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
		// E = E*g + noiseGain*cn, unfused.
		kernMulAdd2(E.Re[:], suppressionGain[:], noiseGain[:], cn.Re[:], E.Re[:])
		kernMulAdd2(E.Im[:], suppressionGain[:], noiseGain[:], cn.Im[:], E.Im[:])

		// Synthesis filterbank.
		const ifftNormalization float32 = 2.0 / FftLength
		// With more than one band the comfort noise of band 1 is synthesized
		// too, and the two inverse transforms run as a pair. The transform
		// reads neither Im[0] nor Im[FftLengthBy2], which is all that
		// FftData::Assign changes in the comfort noise.
		eNoise := &f.fftPair.x0
		if numBands > 1 {
			f.fftPair.Ifft(E, &comfortNoiseHighBand[ch], eExtended, eNoise)
		} else {
			f.fft.Ifft(E, eExtended)
		}

		e0 := e.View(0, ch)
		e0Old := &eOld[0]

		// Overlap-add with the second half of the previous block.
		e0 = e0[:FftLengthBy2]
		kernMulAdd2(e0Old[:], sqrtHanning128[FftLengthBy2:], eExtended[:FftLengthBy2], sqrtHanning128[:FftLengthBy2], e0)
		kernScale(e0, ifftNormalization, e0)

		copy(e0Old[:], eExtended[FftLengthBy2:])

		for b := 1; b < numBands; b++ {
			eBand := e.View(b, ch)
			kernScale(eBand, highBandsGain, eBand)
		}

		if numBands > 1 {
			e1 := e.View(1, ch)
			gain := highBandsNoiseScaling * ifftNormalization
			kernAxpy(eNoise[:FftLengthBy2], gain, e1)
		}

		// The upper bands are delayed to match the filter bank.
		for b := 1; b < numBands; b++ {
			eBand := e.View(b, ch)
			eBandOld := &eOld[b]
			for i := 0; i < FftLengthBy2; i++ {
				eBand[i], eBandOld[i] = eBandOld[i], eBand[i]
			}
		}

		for b := 0; b < numBands; b++ {
			// SafeClamp(x, -32768.f, 32767.f). See kernClampGeneric.
			kernClamp(e.View(b, ch), -32768, 32767)
		}
	}
}
