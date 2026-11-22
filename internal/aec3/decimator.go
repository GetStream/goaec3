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

// Decimator downsamples by 4 or 8 for the delay estimation (decimator.h). It
// holds the sections of its two filters: anti-aliasing, then noise
// reduction.
type Decimator struct {
	downSamplingFactor int
	numAntiAliasing    int
	numNoiseReduction  int
	sections           [len(dlyBandPassFilterDs8) + len(dlyHighPassFilter)]BiQuad
}

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

func (d *Decimator) antiAliasing() []BiQuad { return d.sections[:d.numAntiAliasing] }

func (d *Decimator) noiseReduction() []BiQuad {
	return d.sections[d.numAntiAliasing : d.numAntiAliasing+d.numNoiseReduction]
}

// Decimate downsamples the BlockSize samples of in into the
// BlockSize/downSamplingFactor samples of out.
func (d *Decimator) Decimate(in, out []float32) {
	in = in[:BlockSize]
	out = out[:BlockSize/d.downSamplingFactor]
	if dlyHasDecimateKernel && d.downSamplingFactor == 4 &&
		dlyDecimateDs4((*[BlockSize]float32)(in), (*[BlockSize / 4]float32)(out), (*[4]BiQuad)(d.sections[:4])) {
		return
	}
	var x [BlockSize]float32

	if aa, nr := d.antiAliasing(), d.noiseReduction(); len(aa) == 3 && len(nr) == 1 {
		// Both filters in one pass over the block.
		dlyApplyBiQuad4(in, x[:], &aa[0], &aa[1], &aa[2], &nr[0])
	} else {
		dlyApplyBiQuadCascade(in, x[:], aa)

		if len(nr) > 0 {
			dlyApplyBiQuadCascade(x[:], x[:], nr)
		}
	}

	for j, k := 0, 0; j < len(out); j, k = j+1, k+d.downSamplingFactor {
		out[j] = x[k]
	}
}
