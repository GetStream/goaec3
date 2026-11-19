package aec3

// signal.ellip(6, 1, 40, 1800/8000, 'lowpass', output='sos')
var dlyLowPassFilterDs4 = [3]BiQuadCoefficients{
	{B: [3]float32{0.0180919877, 0.00320961363, 0.0180919877}, A: [2]float32{-1.5183195, 0.633165865}},
	{B: [3]float32{1.0, -1.24550459, 1.0}, A: [2]float32{-1.49784254, 0.853586692}},
	{B: [3]float32{1.0, -1.4221681, 1.0}, A: [2]float32{-1.49791282, 0.969572384}},
}

// signal.cheby1(1, 6, [1000/8000, 2000/8000], 'bandpass', output='sos')
// repeated 5 times.
var dlyBandPassFilterDs8 = [5]BiQuadCoefficients{
	{B: [3]float32{0.103304783, 0.0, -0.103304783}, A: [2]float32{-1.520363, 0.793390435}},
	{B: [3]float32{0.103304783, 0.0, -0.103304783}, A: [2]float32{-1.520363, 0.793390435}},
	{B: [3]float32{0.103304783, 0.0, -0.103304783}, A: [2]float32{-1.520363, 0.793390435}},
	{B: [3]float32{0.103304783, 0.0, -0.103304783}, A: [2]float32{-1.520363, 0.793390435}},
	{B: [3]float32{0.103304783, 0.0, -0.103304783}, A: [2]float32{-1.520363, 0.793390435}},
}

// signal.butter(2, 1000/8000.0, 'highpass', output='sos')
var dlyHighPassFilter = [1]BiQuadCoefficients{
	{B: [3]float32{0.757076375, -1.51415275, 0.757076375}, A: [2]float32{-1.45424359, 0.574061915}},
}

// Decimator provides decimation by 4 or 8 for the purposes of delay
// estimation (decimator.h). The sections of its two cascaded biquad filters
// are stored in the Decimator itself: the anti-aliasing sections followed by
// the noise reduction ones.
type Decimator struct {
	downSamplingFactor int
	numAntiAliasing    int
	numNoiseReduction  int
	sections           [len(dlyBandPassFilterDs8) + len(dlyHighPassFilter)]BiQuad
}

// NewDecimator returns a decimator for downSamplingFactor 4 or 8.
func NewDecimator(downSamplingFactor int) *Decimator {
	d := &Decimator{}
	d.init(downSamplingFactor)
	return d
}

func (d *Decimator) init(downSamplingFactor int) {
	if downSamplingFactor != 4 && downSamplingFactor != 8 {
		panic("aec3: Decimator requires a down-sampling factor of 4 or 8")
	}
	*d = Decimator{downSamplingFactor: downSamplingFactor}
	var antiAliasing, noiseReduction []BiQuadCoefficients
	if downSamplingFactor == 4 {
		antiAliasing, noiseReduction = dlyLowPassFilterDs4[:], dlyHighPassFilter[:]
	} else {
		antiAliasing = dlyBandPassFilterDs8[:]
	}
	d.numAntiAliasing, d.numNoiseReduction = len(antiAliasing), len(noiseReduction)
	for i, c := range antiAliasing {
		d.sections[i] = BiQuad{Coefficients: c}
	}
	for i, c := range noiseReduction {
		d.sections[d.numAntiAliasing+i] = BiQuad{Coefficients: c}
	}
}

// antiAliasing returns the sections of the anti-aliasing filter.
func (d *Decimator) antiAliasing() []BiQuad { return d.sections[:d.numAntiAliasing] }

// noiseReduction returns the sections of the noise reduction filter.
func (d *Decimator) noiseReduction() []BiQuad {
	return d.sections[d.numAntiAliasing : d.numAntiAliasing+d.numNoiseReduction]
}

// Decimate downsamples the BlockSize samples of in into
// BlockSize/downSamplingFactor samples of out.
func (d *Decimator) Decimate(in, out []float32) {
	in = in[:BlockSize]
	out = out[:BlockSize/d.downSamplingFactor]
	var x [BlockSize]float32

	if aa, nr := d.antiAliasing(), d.noiseReduction(); len(aa) == 3 && len(nr) == 1 {
		// Limits the frequency content of the signal to avoid aliasing and
		// reduces the impact of near-end noise, in one pass over the block.
		dlyApplyBiQuad4(in, x[:], &aa[0], &aa[1], &aa[2], &nr[0])
	} else {
		// Limit the frequency content of the signal to avoid aliasing.
		dlyApplyBiQuadCascade(in, x[:], aa)

		// Reduce the impact of near-end noise.
		if len(nr) > 0 {
			dlyApplyBiQuadCascade(x[:], x[:], nr)
		}
	}

	// Downsample the signal.
	for j, k := 0, 0; j < len(out); j, k = j+1, k+d.downSamplingFactor {
		out[j] = x[k]
	}
}
