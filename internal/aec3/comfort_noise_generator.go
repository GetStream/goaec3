package aec3

import "math"

// supGetNoiseFloorFactor computes the noise floor value that matches a WGN
// input of noiseFloorDbfs.
func supGetNoiseFloorFactor(noiseFloorDbfs float32) float32 {
	// supDbfsNormalization = 20*log10(32768).
	const supDbfsNormalization float32 = 90.30899869919436
	return 64 * Exp10f((supDbfsNormalization+noiseFloorDbfs)*0.1)
}

// supSqrt2Sin is the table of sqrt(2) * sin(2*pi*i/32).
var supSqrt2Sin = [32]float32{
	+0.0000000, +0.2758994, +0.5411961, +0.7856950, +1.0000000,
	+1.1758756, +1.3065630, +1.3870398, +float32(math.Sqrt2), +1.3870398,
	+1.3065630, +1.1758756, +1.0000000, +0.7856950, +0.5411961,
	+0.2758994, +0.0000000, -0.2758994, -0.5411961, -0.7856950,
	-1.0000000, -1.1758756, -1.3065630, -1.3870398, -float32(math.Sqrt2),
	-1.3870398, -1.3065630, -1.1758756, -1.0000000, -0.7856950,
	-0.5411961, -0.2758994,
}

func supGenerateComfortNoise(opt Aec3Optimization, N2 *[FftLengthBy2Plus1]float32,
	seed *uint32, nLow, nHigh *FftData) {
	// Compute square root spectrum.
	N := *N2
	NewVectorMath(opt).Sqrt(N[:])

	// Compute the noise level for the upper bands.
	const oneByNumBands float32 = 1.0 / (FftLengthBy2Plus1/2 + 1)
	const fftLengthBy2Plus1By2 = FftLengthBy2Plus1 / 2
	var sum float32
	for _, v := range N[fftLengthBy2Plus1By2:] {
		sum += v
	}
	highBandNoiseLevel := sum * oneByNumBands

	// The analysis and synthesis windowing cause loss of power when
	// cross-fading the noise where frames are completely uncorrelated
	// (generated with random phase), hence the factor sqrt(2).
	//
	// Bin k in [1, FftLengthBy2) draws s = (s*69069 + 1) & (2^31 - 1), a
	// 5-bit index i = s >> 26, x = sqrt(2)*sin(a) = supSqrt2Sin[i] and
	// y = sqrt(2)*cos(a) = supSqrt2Sin[(i+8)&31], and sets
	//
	//	nLow.Re[k], nLow.Im[k] = N[k]*x, N[k]*y                           (spectral shaping)
	//	nHigh.Re[k], nHigh.Im[k] = highBandNoiseLevel*x, highBandNoiseLevel*y  (levelling)
	//
	// The low 31 bits of s after n draws are those of s*69069^n + (69069^(n-1)
	// + ... + 1) in 32-bit arithmetic, so four generators four draws apart
	// produce the draws of bins 4j+1 .. 4j+4 without waiting for each other;
	// the products then run over whole arrays.
	const (
		lcgA  = 69069
		lcgA4 = lcgA * lcgA * lcgA * lcgA & 0xffffffff
		lcgC4 = (lcgA*lcgA*lcgA + lcgA*lcgA + lcgA + 1) & 0xffffffff
	)
	// x[k] and y[k] belong to bin k; bin 0 keeps +0 and bin 64 is drawn but
	// not used.
	var x, y [FftLengthBy2 + 4]float32
	s1 := *seed*lcgA + 1
	s2 := s1*lcgA + 1
	s3 := s2*lcgA + 1
	s4 := s3*lcgA + 1
	for j := 0; ; j += 4 {
		xw := (*[4]float32)(x[j+1 : j+5])
		yw := (*[4]float32)(y[j+1 : j+5])
		i1, i2, i3, i4 := s1>>26&31, s2>>26&31, s3>>26&31, s4>>26&31
		xw[0], yw[0] = supSqrt2Sin[i1], supSqrt2Sin[(i1+8)&31]
		xw[1], yw[1] = supSqrt2Sin[i2], supSqrt2Sin[(i2+8)&31]
		xw[2], yw[2] = supSqrt2Sin[i3], supSqrt2Sin[(i3+8)&31]
		xw[3], yw[3] = supSqrt2Sin[i4], supSqrt2Sin[(i4+8)&31]
		if j == FftLengthBy2-4 {
			break
		}
		s1 = s1*lcgA4 + lcgC4
		s2 = s2*lcgA4 + lcgC4
		s3 = s3*lcgA4 + lcgC4
		s4 = s4*lcgA4 + lcgC4
	}
	// s3 is the draw of bin FftLengthBy2 - 1.
	*seed = s3 & (0x80000000 - 1)

	// The products write bins [0, FftLengthBy2); the C++ leaves Im[0] as it
	// is and sets Re[0] and Re[FftLengthBy2] to zero.
	lowIm0, highIm0 := nLow.Im[0], nHigh.Im[0]
	vectorMultiply(N[:FftLengthBy2], x[:FftLengthBy2], nLow.Re[:FftLengthBy2])
	vectorMultiply(N[:FftLengthBy2], y[:FftLengthBy2], nLow.Im[:FftLengthBy2])
	// highBandNoiseLevel*x: one operand at most is NaN, so the order of the
	// operands does not show.
	kernScale(x[:FftLengthBy2], highBandNoiseLevel, nHigh.Re[:FftLengthBy2])
	kernScale(y[:FftLengthBy2], highBandNoiseLevel, nHigh.Im[:FftLengthBy2])
	nLow.Im[0], nHigh.Im[0] = lowIm0, highIm0
	nLow.Re[0], nLow.Re[FftLengthBy2] = 0, 0
	nHigh.Re[0], nHigh.Re[FftLengthBy2] = 0, 0
}

// ComfortNoiseGenerator generates the comfort noise.
type ComfortNoiseGenerator struct {
	*cngState
	more *cngMore // with more than inlineChannels channels
}

// cngState is the state of a ComfortNoiseGenerator but its pointer.
type cngState struct {
	optimization       Aec3Optimization
	seed               uint32
	numCaptureChannels int
	noiseFloor         float32
	hasN2Initial       bool // false once the initial estimate is released
	n2Initial          [inlineChannels][FftLengthBy2Plus1]float32
	y2Smoothed         [inlineChannels][FftLengthBy2Plus1]float32
	n2                 [inlineChannels][FftLengthBy2Plus1]float32
	n2Counter          int
}

// cngMore is the per-channel state of a ComfortNoiseGenerator with more than
// inlineChannels capture channels.
type cngMore struct {
	n2Initial  [][FftLengthBy2Plus1]float32 // nil once released
	y2Smoothed [][FftLengthBy2Plus1]float32
	n2         [][FftLengthBy2Plus1]float32
}

// bind makes c the view of st.
func (c *ComfortNoiseGenerator) bind(st *cngState, b *binder) {
	c.cngState = st
	if b != nil {
		bindPtr(b, &c.more, &b.x.cngMore)
	}
}

func NewComfortNoiseGenerator(config *EchoCanceller3Config, optimization Aec3Optimization,
	numCaptureChannels int) *ComfortNoiseGenerator {
	c := &ComfortNoiseGenerator{}
	c.bind(&cngState{}, bindAlone)
	c.init(config, optimization, numCaptureChannels)
	return c
}

// init sets up c in place.
func (c *ComfortNoiseGenerator) init(config *EchoCanceller3Config, optimization Aec3Optimization,
	numCaptureChannels int) {
	st := c.cngState
	*c = ComfortNoiseGenerator{}
	c.bind(st, bindAlone)
	*st = cngState{
		optimization:       optimization,
		seed:               42,
		numCaptureChannels: numCaptureChannels,
		noiseFloor:         supGetNoiseFloorFactor(config.ComfortNoise.NoiseFloorDbfs),
		hasN2Initial:       true,
	}
	if numCaptureChannels > inlineChannelLimit {
		c.more = &cngMore{
			n2Initial:  make([][FftLengthBy2Plus1]float32, numCaptureChannels),
			y2Smoothed: make([][FftLengthBy2Plus1]float32, numCaptureChannels),
			n2:         make([][FftLengthBy2Plus1]float32, numCaptureChannels),
		}
	}
	n2 := c.NoiseSpectrum()
	for ch := range n2 {
		for k := range n2[ch] {
			n2[ch][k] = 1.0e6
		}
	}
}

// Compute computes the comfort noise.
func (c ComfortNoiseGenerator) Compute(saturatedCapture bool,
	captureSpectrum [][FftLengthBy2Plus1]float32,
	lowerBandNoise, upperBandNoise []FftData) {
	Y2 := captureSpectrum
	var more cngMore
	if c.more != nil {
		more = *c.more
	}
	n2 := channelState(&c.n2, c.numCaptureChannels, more.n2)
	var n2Initial [][FftLengthBy2Plus1]float32
	if c.hasN2Initial {
		n2Initial = channelState(&c.n2Initial, c.numCaptureChannels, more.n2Initial)
	}

	if !saturatedCapture {
		// Smooth Y2.
		y2Smoothed := channelState(&c.y2Smoothed, c.numCaptureChannels, more.y2Smoothed)
		for ch := range y2Smoothed {
			kernSmooth(y2Smoothed[ch][:], Y2[ch][:], 0.1)
		}

		if c.n2Counter > 50 {
			// Update N2 from Y2_smoothed.
			for ch := range n2 {
				kernNoiseUpdate(&n2[ch], &y2Smoothed[ch])
			}
		}

		if c.hasN2Initial {
			c.n2Counter++
			if c.n2Counter == 1000 {
				c.hasN2Initial = false
				if c.more != nil {
					c.more.n2Initial = nil
				}
			} else {
				// Compute the N2_initial from N2.
				for ch := range n2 {
					kernNoiseInitialUpdate(&n2[ch], &n2Initial[ch])
				}
			}
		}

		for ch := range n2 {
			kernStdMaxScalar(n2[ch][:], c.noiseFloor)
			if c.hasN2Initial {
				kernStdMaxScalar(n2Initial[ch][:], c.noiseFloor)
			}
		}
	}

	// Choose N2 estimate to use.
	N2 := n2
	if c.hasN2Initial {
		N2 = n2Initial
	}

	for ch := range N2 {
		supGenerateComfortNoise(c.optimization, &N2[ch], &c.seed,
			&lowerBandNoise[ch], &upperBandNoise[ch])
	}
}

// NoiseSpectrum returns the estimate of the background noise spectrum.
func (c ComfortNoiseGenerator) NoiseSpectrum() [][FftLengthBy2Plus1]float32 {
	if c.more != nil {
		return c.more.n2
	}
	return c.n2[:c.numCaptureChannels:c.numCaptureChannels]
}
