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
		dlyApplyBiQuadCascade(x, y, f.biquads)
	} else {
		copy(y[:len(x)], x)
	}
}

// ProcessInPlace applies the biquads on y in place (the C++ Process(y)
// overload).
func (f *CascadedBiQuadFilter) ProcessInPlace(y []float32) {
	if len(f.biquads) > 0 {
		dlyApplyBiQuadCascade(y, y, f.biquads)
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

// dlyApplyBiQuadCascade applies the sections in order, x to y for the first
// and in place on y for the rest. Up to four sections run in one pass over
// the samples: each section still receives the output of the previous one
// sample by sample and performs the operations of dlyApplyBiQuad in the same
// order, so the result is the same as applying the sections one after the
// other.
func dlyApplyBiQuadCascade(x, y []float32, biquads []BiQuad) {
	for len(biquads) > 0 {
		switch len(biquads) {
		case 1:
			dlyApplyBiQuad(x, y, &biquads[0])
			return
		case 2:
			dlyApplyBiQuad2(x, y, &biquads[0], &biquads[1])
			return
		case 3:
			dlyApplyBiQuad3(x, y, &biquads[0], &biquads[1], &biquads[2])
			return
		}
		dlyApplyBiQuad4(x, y, &biquads[0], &biquads[1], &biquads[2], &biquads[3])
		biquads = biquads[4:]
		x = y
	}
}

// The multi-section kernels keep the section states in locals and read the
// coefficients from memory (holding them in registers as well spills).

func dlyApplyBiQuad2(x, y []float32, b0, b1 *BiQuad) {
	y = y[:len(x)]
	c0, c1 := &b0.Coefficients, &b1.Coefficients
	ax0, ax1, ay0, ay1 := b0.X[0], b0.X[1], b0.Y[0], b0.Y[1]
	bx0, bx1, by0, by1 := b1.X[0], b1.X[1], b1.Y[0], b1.Y[1]
	for k, v := range x {
		w := float32(c0.B[0]*v) + float32(c0.B[1]*ax0) + float32(c0.B[2]*ax1) - float32(c0.A[0]*ay0) - float32(c0.A[1]*ay1)
		ax1, ax0, ay1, ay0 = ax0, v, ay0, w
		v = w
		w = float32(c1.B[0]*v) + float32(c1.B[1]*bx0) + float32(c1.B[2]*bx1) - float32(c1.A[0]*by0) - float32(c1.A[1]*by1)
		bx1, bx0, by1, by0 = bx0, v, by0, w
		y[k] = w
	}
	b0.X[0], b0.X[1], b0.Y[0], b0.Y[1] = ax0, ax1, ay0, ay1
	b1.X[0], b1.X[1], b1.Y[0], b1.Y[1] = bx0, bx1, by0, by1
}

func dlyApplyBiQuad3(x, y []float32, b0, b1, b2 *BiQuad) {
	y = y[:len(x)]
	c0, c1, c2 := &b0.Coefficients, &b1.Coefficients, &b2.Coefficients
	ax0, ax1, ay0, ay1 := b0.X[0], b0.X[1], b0.Y[0], b0.Y[1]
	bx0, bx1, by0, by1 := b1.X[0], b1.X[1], b1.Y[0], b1.Y[1]
	cx0, cx1, cy0, cy1 := b2.X[0], b2.X[1], b2.Y[0], b2.Y[1]
	for k, v := range x {
		w := float32(c0.B[0]*v) + float32(c0.B[1]*ax0) + float32(c0.B[2]*ax1) - float32(c0.A[0]*ay0) - float32(c0.A[1]*ay1)
		ax1, ax0, ay1, ay0 = ax0, v, ay0, w
		v = w
		w = float32(c1.B[0]*v) + float32(c1.B[1]*bx0) + float32(c1.B[2]*bx1) - float32(c1.A[0]*by0) - float32(c1.A[1]*by1)
		bx1, bx0, by1, by0 = bx0, v, by0, w
		v = w
		w = float32(c2.B[0]*v) + float32(c2.B[1]*cx0) + float32(c2.B[2]*cx1) - float32(c2.A[0]*cy0) - float32(c2.A[1]*cy1)
		cx1, cx0, cy1, cy0 = cx0, v, cy0, w
		y[k] = w
	}
	b0.X[0], b0.X[1], b0.Y[0], b0.Y[1] = ax0, ax1, ay0, ay1
	b1.X[0], b1.X[1], b1.Y[0], b1.Y[1] = bx0, bx1, by0, by1
	b2.X[0], b2.X[1], b2.Y[0], b2.Y[1] = cx0, cx1, cy0, cy1
}

func dlyApplyBiQuad4(x, y []float32, b0, b1, b2, b3 *BiQuad) {
	y = y[:len(x)]
	c0, c1, c2, c3 := &b0.Coefficients, &b1.Coefficients, &b2.Coefficients, &b3.Coefficients
	ax0, ax1, ay0, ay1 := b0.X[0], b0.X[1], b0.Y[0], b0.Y[1]
	bx0, bx1, by0, by1 := b1.X[0], b1.X[1], b1.Y[0], b1.Y[1]
	cx0, cx1, cy0, cy1 := b2.X[0], b2.X[1], b2.Y[0], b2.Y[1]
	dx0, dx1, dy0, dy1 := b3.X[0], b3.X[1], b3.Y[0], b3.Y[1]
	for k, v := range x {
		w := float32(c0.B[0]*v) + float32(c0.B[1]*ax0) + float32(c0.B[2]*ax1) - float32(c0.A[0]*ay0) - float32(c0.A[1]*ay1)
		ax1, ax0, ay1, ay0 = ax0, v, ay0, w
		v = w
		w = float32(c1.B[0]*v) + float32(c1.B[1]*bx0) + float32(c1.B[2]*bx1) - float32(c1.A[0]*by0) - float32(c1.A[1]*by1)
		bx1, bx0, by1, by0 = bx0, v, by0, w
		v = w
		w = float32(c2.B[0]*v) + float32(c2.B[1]*cx0) + float32(c2.B[2]*cx1) - float32(c2.A[0]*cy0) - float32(c2.A[1]*cy1)
		cx1, cx0, cy1, cy0 = cx0, v, cy0, w
		v = w
		w = float32(c3.B[0]*v) + float32(c3.B[1]*dx0) + float32(c3.B[2]*dx1) - float32(c3.A[0]*dy0) - float32(c3.A[1]*dy1)
		dx1, dx0, dy1, dy0 = dx0, v, dy0, w
		y[k] = w
	}
	b0.X[0], b0.X[1], b0.Y[0], b0.Y[1] = ax0, ax1, ay0, ay1
	b1.X[0], b1.X[1], b1.Y[0], b1.Y[1] = bx0, bx1, by0, by1
	b2.X[0], b2.X[1], b2.Y[0], b2.Y[1] = cx0, cx1, cy0, cy1
	b3.X[0], b3.X[1], b3.Y[0], b3.Y[1] = dx0, dx1, dy0, dy1
}
