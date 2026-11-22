//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

// The kern_bins.go helpers without archsimd: the references.

func kernScale(x []float32, a float32, z []float32) { kernScaleGeneric(x, a, z) }

func kernAdd(x, y, z []float32) { kernAddGeneric(x, y, z) }

func kernSub(x, y, z []float32) { kernSubGeneric(x, y, z) }

func kernSumRows(dst *[FftLengthBy2Plus1]float32, rows []*[FftLengthBy2Plus1]float32) {
	kernSumRowsGeneric(dst, rows)
}

func kernUnpack(v *[FftLength]float32, re, im *[FftLengthBy2Plus1]float32) {
	kernUnpackGeneric(v, re, im)
}

func kernPack(re, im *[FftLengthBy2Plus1]float32, v *[FftLength]float32) { kernPackGeneric(re, im, v) }

// kernMinMaxFast reports whether kernMinMax has a fast path.
const kernMinMaxFast = false

// kernMinMax has no fast path: ok is false and the caller runs its loop.
func kernMinMax(x []float32) (mn, mx float32, ok bool) { return 0, 0, false }

func kernClamp(x []float32, lo, hi float32) { kernClampGeneric(x, lo, hi) }

func kernMulAdd2(a, b, c, d, z []float32) { kernMulAdd2Generic(a, b, c, d, z) }

func kernAxpy(x []float32, a float32, z []float32) { kernAxpyGeneric(x, a, z) }

func kernOneMinusSq(x, z []float32) { kernOneMinusSqGeneric(x, z) }

func kernDiffPower(yr, er, yi, ei, z []float32) { kernDiffPowerGeneric(yr, er, yi, ei, z) }

func kernStdMin(x, y, z []float32) { kernStdMinGeneric(x, y, z) }

func kernStdMaxScalar(x []float32, a float32) { kernStdMaxScalarGeneric(x, a) }

func kernSmooth(s, y []float32, c float32) { kernSmoothGeneric(s, y, c) }

func kernNoAudibleEcho(nearend, echo, masker, enrT, enrS, emrT, gain *[FftLengthBy2Plus1]float32) {
	kernNoAudibleEchoGeneric(nearend, echo, masker, enrT, enrS, emrT, gain)
}

func kernMinGain(w *[FftLengthBy2Plus1]float32, minEchoPower float32, minGain *[FftLengthBy2Plus1]float32) {
	kernMinGainGeneric(w, minEchoPower, minGain)
}

func kernWeigh(threshold, normalizer float32, echo, weightedEcho []float32) {
	kernWeighGeneric(threshold, normalizer, echo, weightedEcho)
}

func kernNoiseUpdate(n, s *[FftLengthBy2Plus1]float32) { kernNoiseUpdateGeneric(n, s) }

func kernNoiseInitialUpdate(n, ini *[FftLengthBy2Plus1]float32) {
	kernNoiseInitialUpdateGeneric(n, ini)
}

func kernFir3(x []float32, h *[3]float32, out []float32, from, to int) {
	kernFir3Generic(x, h, out, from, to)
}

func kernDiv(x, y, z []float32) { kernDivGeneric(x, y, z) }

func kernReverb(r, p, s []float32, d float32) { kernReverbGeneric(r, p, s, d) }

func kernStdMax(x, y, z []float32) { kernStdMaxGeneric(x, y, z) }

func kernRefinedMu(X2, hError, E2 *[FftLengthBy2Plus1]float32, gate, sp float32, mu *[FftLengthBy2Plus1]float32) {
	kernRefinedMuGeneric(X2, hError, E2, gate, sp, mu)
}

func kernRefinedHError(mu, X2, hError *[FftLengthBy2Plus1]float32) {
	kernRefinedHErrorGeneric(mu, X2, hError)
}

func kernRefinedLeak(E2Refined, E2Coarse *[FftLengthBy2Plus1]float32, erl []float32, disallowLeakageDiverged bool,
	leakageConverged, leakageDiverged, errorFloor, errorCeil float32, hError *[FftLengthBy2Plus1]float32) {
	kernRefinedLeakGeneric(E2Refined, E2Coarse, erl, disallowLeakageDiverged, leakageConverged, leakageDiverged,
		errorFloor, errorCeil, hError)
}

func kernCoarseMu(X2 *[FftLengthBy2Plus1]float32, gate, rate float32, mu *[FftLengthBy2Plus1]float32) {
	kernCoarseMuGeneric(X2, gate, rate, mu)
}

func kernReverbScalar(r, p []float32, s, d float32) { kernReverbScalarGeneric(r, p, s, d) }

func kernErlUpdate(X2, Y2, erl *[FftLengthBy2Plus1]float32, hold *[FftLengthBy2]int32) {
	kernErlUpdateGeneric(X2, Y2, erl, hold)
}

func kernPeakBin(x *[FftLengthBy2Plus1]float32) int { return kernPeakBinGeneric(x) }

func kernNarrowBands(x *[FftLengthBy2Plus1]float32, narrow *[FftLengthBy2]int32) {
	kernNarrowBandsGeneric(x, narrow)
}

func kernNoiseFloorUpdate(renderPower, noiseFloor *[FftLengthBy2Plus1]float32, counter *[FftLengthBy2Plus1]int32,
	hold int32, minPower float32) {
	kernNoiseFloorUpdateGeneric(renderPower, noiseFloor, counter, hold, minPower)
}

func kernStdMinScalar(x []float32, a float32) { kernStdMinScalarGeneric(x, a) }

func kernSumMaxSq(x *[BlockSize]float32, sum, mx float32) (float32, float32) {
	return kernSumMaxSqGeneric(x, sum, mx)
}
