//go:build goexperiment.simd && arm64 && !purego

package bands

func bsAnalysisKernel(hist *[bsSubSampling][bsHistSize]float32, in *[FullBandSize]float32,
	out0, out1, out2 *[SplitBandSize]float32) {
	bsAnalysisSIMD(hist, in, out0, out1, out2)
}

func bsSynthesisKernel(hist *[bsNumSynthesisInputs][bsHistSize]float32,
	in0, in1, in2 *[SplitBandSize]float32, out *[FullBandSize]float32) {
	bsSynthesisSIMD(hist, in0, in1, in2, out)
}
