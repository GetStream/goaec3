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

// Decimator decimates a block of samples (decimator.h).
type Decimator struct {
	downSamplingFactor   int
	antiAliasingFilter   CascadedBiQuadFilter
	noiseReductionFilter CascadedBiQuadFilter
}

// NewDecimator returns a decimator for a down-sampling factor of 4 or 8.
func NewDecimator(downSamplingFactor int) *Decimator {
	d := &Decimator{}
	d.init(downSamplingFactor)
	return d
}

func (d *Decimator) init(downSamplingFactor int) {
	if downSamplingFactor != 4 && downSamplingFactor != 8 {
		panic("aec3: Decimator requires a down-sampling factor of 4 or 8")
	}
	d.downSamplingFactor = downSamplingFactor
	if downSamplingFactor == 4 {
		d.antiAliasingFilter.init(dlyLowPassFilterDs4[:])
		d.noiseReductionFilter.init(dlyHighPassFilter[:])
	} else {
		d.antiAliasingFilter.init(dlyBandPassFilterDs8[:])
		d.noiseReductionFilter.init(nil)
	}
}

// Decimate downsamples the BlockSize samples of in into out, which holds
// BlockSize/downSamplingFactor samples.
func (d *Decimator) Decimate(in, out []float32) {
	in = in[:BlockSize]
	out = out[:BlockSize/d.downSamplingFactor]
	var x [BlockSize]float32

	if aa, nr := d.antiAliasingFilter.biquads, d.noiseReductionFilter.biquads; len(aa) == 3 && len(nr) == 1 {
		// Both filters in one pass over the samples (see
		// dlyApplyBiQuadCascade).
		dlyApplyBiQuad4(in, x[:], &aa[0], &aa[1], &aa[2], &nr[0])
	} else {
		// Limit the frequency content of the signal to avoid aliasing.
		d.antiAliasingFilter.Process(in, x[:])

		// Reduce the impact of near-end noise.
		d.noiseReductionFilter.ProcessInPlace(x[:])
	}

	// Downsample the signal.
	for j, k := 0, 0; j < len(out); j, k = j+1, k+d.downSamplingFactor {
		out[j] = x[k]
	}
}
