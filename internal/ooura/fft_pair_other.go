//go:build !(goexperiment.simd && amd64 && !purego)

package ooura

// PairScratch is the work area of FftPair and InverseFftPair; this build
// runs the two transforms one after the other and needs none.
type PairScratch struct{}

func fftPair(a, b *[128]float32, _ *PairScratch) {
	fft(a)
	fft(b)
}

func inverseFftPair(a, b *[128]float32, _ *PairScratch) {
	inverseFft(a)
	inverseFft(b)
}

func initPairTables() {}
