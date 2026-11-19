//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

// Without archsimd every MatchedFilterCore lane runs its pure-Go reference.

func dlyDotSNeon(h, x []float32, acc *[4]float32) { dlyDotSNeonRef(h, x, acc) }

func dlyAxpyDotSNeon(h, xa, xb []float32, alpha float32, acc *[4]float32) {
	dlyAxpyDotSNeonRef(h, xa, xb, alpha, acc)
}

func dlySqNeon(x []float32, acc *[4]float32) { dlySqNeonRef(x, acc) }

func dlyDotSSse2(h, x []float32, acc *[8]float32) { dlyDotSSse2Ref(h, x, acc) }

func dlyAxpyDotSSse2(h, xa, xb []float32, alpha float32, acc *[8]float32) {
	dlyAxpyDotSSse2Ref(h, xa, xb, alpha, acc)
}

func dlySqSse2(x []float32, acc *[8]float32) { dlySqSse2Ref(x, acc) }

func dlyDotSAvx2(h, x []float32, acc *[16]float32) { dlyDotSAvx2Ref(h, x, acc) }

func dlyAxpyFmaDotSAvx2(h, xa, xb []float32, alpha float32, acc *[16]float32) {
	dlyAxpyFmaDotSAvx2Ref(h, xa, xb, alpha, acc)
}

func dlySqAvx2(x []float32, acc *[16]float32) { dlySqAvx2Ref(x, acc) }

func dlyDotSSumSse2(h, x []float32) float32 { return dlyDotSSumSse2Ref(h, x) }

func dlyAxpyDotSSumSse2(h, xa, xb []float32, alpha float32) float32 {
	return dlyAxpyDotSSumSse2Ref(h, xa, xb, alpha)
}

func dlyDotSSumAvx2(h, x []float32) float32 { return dlyDotSSumAvx2Ref(h, x) }

func dlyAxpyFmaDotSSumAvx2(h, xa, xb []float32, alpha float32) float32 {
	return dlyAxpyFmaDotSSumAvx2Ref(h, xa, xb, alpha)
}

func dlyX2Lanes(g int, x []float32, n int, lanes []float32) { dlyX2LanesRef(g, x, n, lanes) }

// dlySqStreams is the number of stretches dlySqMulti accumulates at once.
// Without archsimd the multi-stretch routines are not used; three stretches
// keep them testable.
func dlySqStreams(g int) int { return 3 }

func dlySqMulti(g int, x []float32, n int, offs *[dlyMaxStreams]int, acc *[dlyMaxStreams][16]float32) {
	dlySqMultiRef(g, x, n, offs, acc, 3)
}

func dlyX2Wrapped(g int, x []float32, hSize, lo, r int, out *[dlyX2Run]float32) {
	dlyX2WrappedRef(g, x, hSize, lo, r, out)
}

func dlyAccErrNeon(h, x []float32, y float32, accErr []float32) float32 {
	return dlyAccErrNeonRef(h, x, y, accErr)
}

func dlyAxpyAccErrNeon(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	return dlyAxpyAccErrNeonRef(h, xa, xb, alpha, y, accErr)
}

func dlyAccErrSse2(h, x []float32, y float32, accErr []float32) float32 {
	return dlyAccErrSse2Ref(h, x, y, accErr)
}

func dlyAxpyAccErrSse2(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	return dlyAxpyAccErrSse2Ref(h, xa, xb, alpha, y, accErr)
}

func dlyAccErrAvx2(h, x []float32, y float32, accErr []float32) float32 {
	return dlyAccErrAvx2Ref(h, x, y, accErr)
}

func dlyAxpyFmaAccErrAvx2(h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	return dlyAxpyFmaAccErrAvx2Ref(h, xa, xb, alpha, y, accErr)
}

func dlyX2Reduce(g int, accErr bool, lanes *[dlyX2Run + 15]float32, r int, out *[dlyX2Run]float32) {
	dlyX2ReduceRef(g, accErr, lanes, r, out)
}

func dlyAxpy(h, x []float32, alpha float32) { dlyAxpyRef(h, x, alpha) }

func dlyAxpyFma(h, x []float32, alpha float32) { dlyAxpyFmaRef(h, x, alpha) }

// dlyPairNeon selects the two-filter kernels for the NEON lane; they exist
// as archsimd kernels only.
var dlyPairNeon = false

func dlyDotS2Neon(hA, xA []float32, accA *[4]float32, hB, xB []float32, accB *[4]float32) {
	dlyDotS2NeonRef(hA, xA, accA, hB, xB, accB)
}

func dlyAxpyDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[4]float32) {
	dlyAxpyDotS2NeonRef(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
}

func dlyMaxSquarePeakIndex(h []float32) int { return dlyMaxSquarePeakIndexRef(h) }

// dlyZeroAcc clears the lane accumulators.
func dlyZeroAcc(acc *[16]float32) { *acc = [16]float32{} }

// dlyPairSse2 and dlyPairAvx2 select the two-filter kernels for the SSE2
// and AVX2 lanes; they exist as archsimd kernels on amd64 only.
var dlyPairSse2, dlyPairAvx2 = false, false

func dlyDotS2Sse2(hA, xA []float32, accA *[8]float32, hB, xB []float32, accB *[8]float32) {
	dlyDotS2Sse2Ref(hA, xA, accA, hB, xB, accB)
}

func dlyAxpyDotS2Sse2(hA, xaA, xbA []float32, alphaA float32, accA *[8]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[8]float32) {
	dlyAxpyDotS2Sse2Ref(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
}

func dlyDotS2Avx2(hA, xA []float32, accA *[16]float32, hB, xB []float32, accB *[16]float32) {
	dlyDotS2Avx2Ref(hA, xA, accA, hB, xB, accB)
}

func dlyAxpyFmaDotS2Avx2(hA, xaA, xbA []float32, alphaA float32, accA *[16]float32,
	hB, xaB, xbB []float32, alphaB float32, accB *[16]float32) {
	dlyAxpyFmaDotS2Avx2Ref(hA, xaA, xbA, alphaA, accA, hB, xaB, xbB, alphaB, accB)
}

func dlyAxpyDotDotS2Neon(hA, xaA, xbA []float32, alphaA float32, accA *[4]float32,
	hB, xB []float32, accB *[4]float32) {
	dlyAxpyDotDotS2NeonRef(hA, xaA, xbA, alphaA, accA, hB, xB, accB)
}

func dlyX2Lanes2(g int, xA, xB []float32, n int, lanesA, lanesB []float32) {
	dlyX2Lanes2Ref(g, xA, xB, n, lanesA, lanesB)
}

// dlyDotS2Sum and dlyAxpyDotS2Sum are the two-filter Sum kernels of the
// lane with group size g: dlyDotSSum (dlyAxpyDotSSum) for two filters at
// once, with the reduced sums before their addition to the zero tail in
// out[0] and out[2]. They report false where the lane has none.

func dlyDotS2Sum(g int, hA, xA, hB, xB []float32, out *[4]float32) bool { return false }

func dlyAxpyDotS2Sum(g int, hA, xaA, xbA []float32, alphaA float32,
	hB, xaB, xbB []float32, alphaB float32, out *[4]float32) bool {
	return false
}
