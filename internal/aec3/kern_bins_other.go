//go:build !goexperiment.simd || purego || !arm64

package aec3

// kern_bins.go helpers that have archsimd kernels on arm64 only.

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
