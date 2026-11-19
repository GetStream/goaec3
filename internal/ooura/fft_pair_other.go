//go:build !(goexperiment.simd && amd64 && !purego)

package ooura

// PairScratch is the work area of FftSplitPair and InverseFftSplitPair; this
// build runs the two transforms one after the other and needs none.
type PairScratch struct{}

func fftSplitPair(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, _ *PairScratch) {
	fftSplit(a, re0, im0)
	fftSplit(b, re1, im1)
}

func inverseFftSplitPair(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, _ *PairScratch) {
	inverseFftSplit(re0, im0, a)
	inverseFftSplit(re1, im1, b)
}

func initPairTables() {}
