package aec3

// Port of adaptive_fir_filter_erl_avx2.cc.

// ErlComputerAVX2 is aec3::ErlComputer_AVX2.
func ErlComputerAVX2(H2 [][FftLengthBy2Plus1]float32, erl []float32) {
	afirErlComputerSimd(OptAvx2, H2, erl)
}
