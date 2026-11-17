package aec3

// BiQuadCoefficients mirrors CascadedBiQuadFilter::BiQuadCoefficients.
type BiQuadCoefficients struct {
	B [3]float32
	A [2]float32
}

// BiQuad mirrors CascadedBiQuadFilter::BiQuad: one direct form 1 section
// with its state.
type BiQuad struct {
	Coefficients BiQuadCoefficients
	X            [2]float32
	Y            [2]float32
}

// Reset clears the section state.
func (b *BiQuad) Reset() {
	b.X = [2]float32{}
	b.Y = [2]float32{}
}

// CascadedBiQuadFilter applies a number of biquads in a cascaded manner
// (modules/audio_processing/utility/cascaded_biquad_filter.h). The filter
// implementation is direct form 1.
type CascadedBiQuadFilter struct {
	biquads []BiQuad
}

// NewCascadedBiQuadFilter returns a filter with one section per coefficient
// set.
func NewCascadedBiQuadFilter(coefficients []BiQuadCoefficients) *CascadedBiQuadFilter {
	f := &CascadedBiQuadFilter{}
	f.init(coefficients)
	return f
}

func (f *CascadedBiQuadFilter) init(coefficients []BiQuadCoefficients) {
	f.biquads = make([]BiQuad, len(coefficients))
	for i, c := range coefficients {
		f.biquads[i] = BiQuad{Coefficients: c}
	}
}

// Process applies the biquads on x to form the output in y.
func (f *CascadedBiQuadFilter) Process(x, y []float32) {
	if len(f.biquads) > 0 {
		dlyApplyBiQuad(x, y, &f.biquads[0])
		for k := 1; k < len(f.biquads); k++ {
			dlyApplyBiQuad(y, y, &f.biquads[k])
		}
	} else {
		copy(y[:len(x)], x)
	}
}

// ProcessInPlace applies the biquads on y in place (the C++ Process(y)
// overload).
func (f *CascadedBiQuadFilter) ProcessInPlace(y []float32) {
	for k := range f.biquads {
		dlyApplyBiQuad(y, y, &f.biquads[k])
	}
}

// Reset resets the filter to its initial state.
func (f *CascadedBiQuadFilter) Reset() {
	for k := range f.biquads {
		f.biquads[k].Reset()
	}
}

func dlyApplyBiQuad(x, y []float32, biquad *BiQuad) {
	y = y[:len(x)]
	cA0 := biquad.Coefficients.A[0]
	cA1 := biquad.Coefficients.A[1]
	cB0 := biquad.Coefficients.B[0]
	cB1 := biquad.Coefficients.B[1]
	cB2 := biquad.Coefficients.B[2]
	mX0 := biquad.X[0]
	mX1 := biquad.X[1]
	mY0 := biquad.Y[0]
	mY1 := biquad.Y[1]
	for k := range x {
		tmp := x[k]
		y[k] = float32(cB0*tmp) + float32(cB1*mX0) + float32(cB2*mX1) - float32(cA0*mY0) -
			float32(cA1*mY1)
		mX1 = mX0
		mX0 = tmp
		mY1 = mY0
		mY0 = y[k]
	}
	biquad.X[0] = mX0
	biquad.X[1] = mX1
	biquad.Y[0] = mY0
	biquad.Y[1] = mY1
}
