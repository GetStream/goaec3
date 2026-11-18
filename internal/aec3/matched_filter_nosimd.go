//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

// Without archsimd every MatchedFilterCore lane runs its pure-Go reference.

func dlyDotNeon(h, x []float32, acc *[8]float32) { dlyDotNeonRef(h, x, acc) }

func dlyDotSse2(h, x []float32, acc *[16]float32) { dlyDotSse2Ref(h, x, acc) }

func dlyDotAvx2(h, x []float32, acc *[32]float32) { dlyDotAvx2Ref(h, x, acc) }

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32, x2 *[4]float32) float32 {
	return dlyAccErrNeonRef(h, x, y, accErr, x2)
}

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32, x2 *[8]float32) float32 {
	return dlyAccErrSse2Ref(h, x, y, accErr, x2)
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32, x2 *[16]float32) float32 {
	return dlyAccErrAvx2Ref(h, x, y, accErr, x2)
}

func dlyAxpy(h, x []float32, alpha float32) { dlyAxpyRef(h, x, alpha) }

func dlyAxpyFma(h, x []float32, alpha float32) { dlyAxpyFmaRef(h, x, alpha) }

func dlyAxpyDotNeon(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	dlyAxpyDotNeonRef(h, xa, xb, alpha, acc)
}

func dlyAxpyDotSse2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	dlyAxpyDotSse2Ref(h, xa, xb, alpha, acc)
}

func dlyAxpyFmaDotAvx2(h, xa, xb []float32, alpha float32, acc *[32]float32) {
	dlyAxpyFmaDotAvx2Ref(h, xa, xb, alpha, acc)
}

func dlyMaxSquarePeakIndex(h []float32) int { return dlyMaxSquarePeakIndexRef(h) }

// dlyCoreLane runs the MatchedFilterCore loop of the SIMD lane with group
// size g, with every update fused into the next dot product.
func dlyCoreLane(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	dlyCoreFused(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
}
