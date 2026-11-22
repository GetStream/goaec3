//go:build !(goexperiment.simd && amd64 && !purego)

package ooura

// PairScratch is the work area of FftSplitPair and InverseFftSplitPair; this
// build runs the two transforms one after the other.
type PairScratch struct {
	s Scratch
}

func fftSplitPair(a, b *[128]float32, re0, im0, re1, im1 *[65]float32, s *PairScratch) {
	fftSplit(a, re0, im0, s.single())
	fftSplit(b, re1, im1, s.single())
}

func inverseFftSplitPair(re0, im0, re1, im1 *[65]float32, a, b *[128]float32, s *PairScratch) {
	inverseFftSplit(re0, im0, a, s.single())
	inverseFftSplit(re1, im1, b, s.single())
}
