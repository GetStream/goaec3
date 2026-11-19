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
	nLow.Re[0], nLow.Re[FftLengthBy2] = 0, 0
	nHigh.Re[0], nHigh.Re[FftLengthBy2] = 0, 0
	s := *seed
	for k := 1; k < FftLengthBy2; k++ {
		const indexMask = 32 - 1
		// Generate a random 31-bit integer.
		s = (s*69069 + 1) & (0x80000000 - 1)
		// Convert to a 5-bit index.
		i := s >> 26

		// y = sqrt(2) * sin(a)
		x := supSqrt2Sin[i]
		// x = sqrt(2) * cos(a) = sqrt(2) * sin(a + pi/2)
		y := supSqrt2Sin[(i+8)&indexMask]

		// Form low-frequency noise via spectral shaping.
		nLow.Re[k] = N[k] * x
		nLow.Im[k] = N[k] * y

		// Form the high-frequency noise via simple levelling.
		nHigh.Re[k] = highBandNoiseLevel * x
		nHigh.Im[k] = highBandNoiseLevel * y
	}
	*seed = s
}

// ComfortNoiseGenerator generates the comfort noise.
type ComfortNoiseGenerator struct {
	more               *cngMore // with more than inlineChannels channels
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

func NewComfortNoiseGenerator(config *EchoCanceller3Config, optimization Aec3Optimization,
	numCaptureChannels int) *ComfortNoiseGenerator {
	c := &ComfortNoiseGenerator{}
	c.init(config, optimization, numCaptureChannels)
	return c
}

// init sets up c in place.
func (c *ComfortNoiseGenerator) init(config *EchoCanceller3Config, optimization Aec3Optimization,
	numCaptureChannels int) {
	*c = ComfortNoiseGenerator{
		optimization:       optimization,
		seed:               42,
		numCaptureChannels: numCaptureChannels,
		noiseFloor:         supGetNoiseFloorFactor(config.ComfortNoise.NoiseFloorDbfs),
		hasN2Initial:       true,
	}
	if numCaptureChannels > inlineChannels {
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
func (c *ComfortNoiseGenerator) Compute(saturatedCapture bool,
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
func (c *ComfortNoiseGenerator) NoiseSpectrum() [][FftLengthBy2Plus1]float32 {
	if c.more != nil {
		return c.more.n2
	}
	return c.n2[:c.numCaptureChannels:c.numCaptureChannels]
}
