package aec3

// Port of adaptive_fir_filter_erl.{h,cc}. Every lane adds the partition
// responses bin by bin in partition order, so all lanes are bit-identical;
// the SIMD lanes route bins 0..63 through afirErlBlock.

// ErlComputer is aec3::ErlComputer (scalar): erl = sum over partitions of H2.
func ErlComputer(H2 [][FftLengthBy2Plus1]float32, erl []float32) {
	erl = erl[:FftLengthBy2Plus1]
	clear(erl)
	for j := range H2 {
		for k := range erl {
			erl[k] += H2[j][k]
		}
	}
}

// ErlComputerNEON is aec3::ErlComputer_NEON.
func ErlComputerNEON(H2 [][FftLengthBy2Plus1]float32, erl []float32) {
	afirErlComputerSimd(OptNeon, H2, erl)
}

// ErlComputerSSE2 is aec3::ErlComputer_SSE2.
func ErlComputerSSE2(H2 [][FftLengthBy2Plus1]float32, erl []float32) {
	afirErlComputerSimd(OptSse2, H2, erl)
}

func afirErlComputerSimd(opt Aec3Optimization, H2 [][FftLengthBy2Plus1]float32, erl []float32) {
	e := (*[FftLengthBy2Plus1]float32)(erl)
	if len(H2) <= rbMaxRows {
		var rows [rbMaxRows]*[FftLengthBy2Plus1]float32
		for j := range H2 {
			rows[j] = &H2[j]
		}
		kernSumRows(e, rows[:len(H2)])
		return
	}
	*e = [FftLengthBy2Plus1]float32{}
	for j := range H2 {
		afirErlBlock(opt, &H2[j], e)
		e[FftLengthBy2] += H2[j][FftLengthBy2]
	}
}

// afirErlBlockGo adds bins 0..63 of h2 into erl.
func afirErlBlockGo(h2, erl *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k++ {
		erl[k] += h2[k]
	}
}

// ComputeErl computes the echo return loss from a frequency response.
func ComputeErl(optimization Aec3Optimization, H2 [][FftLengthBy2Plus1]float32, erl []float32) {
	switch optimization {
	case OptSse2:
		ErlComputerSSE2(H2, erl)
	case OptAvx2:
		ErlComputerAVX2(H2, erl)
	case OptNeon:
		ErlComputerNEON(H2, erl)
	default:
		ErlComputer(H2, erl)
	}
}
