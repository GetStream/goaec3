package aec3

import "github.com/GetStream/goaec3/internal/ooura"

// Window selects the analysis window of an Aec3Fft.
type Window int

const (
	WindowRectangular Window = iota
	WindowHanning
	WindowSqrtHanning
)

var hanning64 = [FftLengthBy2]float32{
	0., 0.00248461, 0.00991376, 0.0222136, 0.03926189,
	0.06088921, 0.08688061, 0.11697778, 0.15088159, 0.1882551,
	0.22872687, 0.27189467, 0.31732949, 0.36457977, 0.41317591,
	0.46263495, 0.51246535, 0.56217185, 0.61126047, 0.65924333,
	0.70564355, 0.75, 0.79187184, 0.83084292, 0.86652594,
	0.89856625, 0.92664544, 0.95048443, 0.96984631, 0.98453864,
	0.99441541, 0.99937846, 0.99937846, 0.99441541, 0.98453864,
	0.96984631, 0.95048443, 0.92664544, 0.89856625, 0.86652594,
	0.83084292, 0.79187184, 0.75, 0.70564355, 0.65924333,
	0.61126047, 0.56217185, 0.51246535, 0.46263495, 0.41317591,
	0.36457977, 0.31732949, 0.27189467, 0.22872687, 0.1882551,
	0.15088159, 0.11697778, 0.08688061, 0.06088921, 0.03926189,
	0.0222136, 0.00991376, 0.00248461, 0.,
}

var sqrtHanning128 = [FftLength]float32{
	0.00000000000000, 0.02454122852291, 0.04906767432742, 0.07356456359967,
	0.09801714032956, 0.12241067519922, 0.14673047445536, 0.17096188876030,
	0.19509032201613, 0.21910124015687, 0.24298017990326, 0.26671275747490,
	0.29028467725446, 0.31368174039889, 0.33688985339222, 0.35989503653499,
	0.38268343236509, 0.40524131400499, 0.42755509343028, 0.44961132965461,
	0.47139673682600, 0.49289819222978, 0.51410274419322, 0.53499761988710,
	0.55557023301960, 0.57580819141785, 0.59569930449243, 0.61523159058063,
	0.63439328416365, 0.65317284295378, 0.67155895484702, 0.68954054473707,
	0.70710678118655, 0.72424708295147, 0.74095112535496, 0.75720884650648,
	0.77301045336274, 0.78834642762661, 0.80320753148064, 0.81758481315158,
	0.83146961230255, 0.84485356524971, 0.85772861000027, 0.87008699110871,
	0.88192126434835, 0.89322430119552, 0.90398929312344, 0.91420975570353,
	0.92387953251129, 0.93299279883474, 0.94154406518302, 0.94952818059304,
	0.95694033573221, 0.96377606579544, 0.97003125319454, 0.97570213003853,
	0.98078528040323, 0.98527764238894, 0.98917650996478, 0.99247953459871,
	0.99518472667220, 0.99729045667869, 0.99879545620517, 0.99969881869620,
	1.00000000000000, 0.99969881869620, 0.99879545620517, 0.99729045667869,
	0.99518472667220, 0.99247953459871, 0.98917650996478, 0.98527764238894,
	0.98078528040323, 0.97570213003853, 0.97003125319454, 0.96377606579544,
	0.95694033573221, 0.94952818059304, 0.94154406518302, 0.93299279883474,
	0.92387953251129, 0.91420975570353, 0.90398929312344, 0.89322430119552,
	0.88192126434835, 0.87008699110871, 0.85772861000027, 0.84485356524971,
	0.83146961230255, 0.81758481315158, 0.80320753148064, 0.78834642762661,
	0.77301045336274, 0.75720884650648, 0.74095112535496, 0.72424708295147,
	0.70710678118655, 0.68954054473707, 0.67155895484702, 0.65317284295378,
	0.63439328416365, 0.61523159058063, 0.59569930449243, 0.57580819141785,
	0.55557023301960, 0.53499761988710, 0.51410274419322, 0.49289819222978,
	0.47139673682600, 0.44961132965461, 0.42755509343028, 0.40524131400499,
	0.38268343236509, 0.35989503653499, 0.33688985339222, 0.31368174039889,
	0.29028467725446, 0.26671275747490, 0.24298017990326, 0.21910124015687,
	0.19509032201613, 0.17096188876030, 0.14673047445536, 0.12241067519922,
	0.09801714032956, 0.07356456359967, 0.04906767432742, 0.02454122852291,
}

// Aec3Fft is the 128-point Ooura real FFT in the lane of the build. The zero
// value is ready to use. An Aec3Fft holds the work area of its transforms
// and serves one goroutine at a time.
type Aec3Fft struct {
	scratch ooura.Scratch
	x       [FftLength]float32
}

// Fft transforms x into X, using x as work space. It is the transform
// followed by FftData::CopyFromPackedArray.
func (f *Aec3Fft) Fft(x *[FftLength]float32, X *FftData) {
	ooura.FftSplit(x, &X.Re, &X.Im, &f.scratch)
}

// Ifft sets x to the inverse transform of X, unscaled. It is
// FftData::CopyToPackedArray followed by the transform.
func (f *Aec3Fft) Ifft(X *FftData, x *[FftLength]float32) {
	ooura.InverseFftSplit(&X.Re, &X.Im, x, &f.scratch)
}

// ZeroPaddedFft transforms the 64 samples of x, windowed, behind 64 zeros.
func (f *Aec3Fft) ZeroPaddedFft(x []float32, window Window, X *FftData) {
	fft := &f.x
	clear(fft[:FftLengthBy2])
	x = x[:FftLengthBy2]
	switch window {
	case WindowRectangular:
		copy(fft[FftLengthBy2:], x)
	case WindowHanning:
		vectorMultiply(x, hanning64[:], fft[FftLengthBy2:])
	default:
		panic("aec3: unsupported window for ZeroPaddedFft")
	}
	f.Fft(fft, X)
}

// PaddedFft transforms the windowed concatenation of xOld and x.
func (f *Aec3Fft) PaddedFft(x, xOld []float32, window Window, X *FftData) {
	fft := &f.x
	x = x[:FftLengthBy2]
	xOld = xOld[:FftLengthBy2]
	switch window {
	case WindowRectangular:
		copy(fft[:], xOld)
		copy(fft[FftLengthBy2:], x)
	case WindowSqrtHanning:
		vectorMultiply(xOld, sqrtHanning128[:FftLengthBy2], fft[:FftLengthBy2])
		vectorMultiply(x, sqrtHanning128[FftLengthBy2:], fft[FftLengthBy2:])
	default:
		panic("aec3: unsupported window for PaddedFft")
	}
	f.Fft(fft, X)
}

// aec3FftPair runs two transforms of Aec3Fft at a time. Each result is that
// of the single transform bit for bit. The zero value is ready to use.
type aec3FftPair struct {
	scratch ooura.PairScratch
	x0, x1  [FftLength]float32
}

// Fft is Aec3Fft.Fft for (x0, X0) and (x1, X1).
func (p *aec3FftPair) Fft(x0, x1 *[FftLength]float32, X0, X1 *FftData) {
	ooura.FftSplitPair(x0, x1, &X0.Re, &X0.Im, &X1.Re, &X1.Im, &p.scratch)
}

// Ifft is Aec3Fft.Ifft for (X0, x0) and (X1, x1).
func (p *aec3FftPair) Ifft(X0, X1 *FftData, x0, x1 *[FftLength]float32) {
	ooura.InverseFftSplitPair(&X0.Re, &X0.Im, &X1.Re, &X1.Im, x0, x1, &p.scratch)
}

// ZeroPaddedFft is Aec3Fft.ZeroPaddedFft with the Hanning window for
// (x0, X0) and (x1, X1).
func (p *aec3FftPair) ZeroPaddedFft(x0, x1 []float32, X0, X1 *FftData) {
	clear(p.x0[:FftLengthBy2])
	clear(p.x1[:FftLengthBy2])
	vectorMultiply(x0[:FftLengthBy2], hanning64[:], p.x0[FftLengthBy2:])
	vectorMultiply(x1[:FftLengthBy2], hanning64[:], p.x1[FftLengthBy2:])
	p.Fft(&p.x0, &p.x1, X0, X1)
}

// PaddedFft is Aec3Fft.PaddedFft with the square root Hanning window for
// (x0, xOld0, X0) and (x1, xOld1, X1).
func (p *aec3FftPair) PaddedFft(x0, xOld0, x1, xOld1 []float32, X0, X1 *FftData) {
	vectorMultiply(xOld0[:FftLengthBy2], sqrtHanning128[:FftLengthBy2], p.x0[:FftLengthBy2])
	vectorMultiply(x0[:FftLengthBy2], sqrtHanning128[FftLengthBy2:], p.x0[FftLengthBy2:])
	vectorMultiply(xOld1[:FftLengthBy2], sqrtHanning128[:FftLengthBy2], p.x1[:FftLengthBy2])
	vectorMultiply(x1[:FftLengthBy2], sqrtHanning128[FftLengthBy2:], p.x1[FftLengthBy2:])
	p.Fft(&p.x0, &p.x1, X0, X1)
}

// PaddedFftRectangular is Aec3Fft.PaddedFft with the rectangular window for
// (x0, xOld0, X0) and (x1, xOld1, X1).
func (p *aec3FftPair) PaddedFftRectangular(x0, xOld0, x1, xOld1 []float32, X0, X1 *FftData) {
	copy(p.x0[:FftLengthBy2], xOld0[:FftLengthBy2])
	copy(p.x0[FftLengthBy2:], x0[:FftLengthBy2])
	copy(p.x1[:FftLengthBy2], xOld1[:FftLengthBy2])
	copy(p.x1[FftLengthBy2:], x1[:FftLengthBy2])
	p.Fft(&p.x0, &p.x1, X0, X1)
}
