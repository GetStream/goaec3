//go:build goexperiment.simd && amd64 && !purego

package bands

import "testing"

// TestAVXOnlyKernels runs the kernel checks with the 128-bit paths that
// CPUs with AVX but not AVX2 take.
func TestAVXOnlyKernels(t *testing.T) {
	if !bsHasSIMD || !bsHasAVX2 {
		t.Skip("no AVX2 to turn off")
	}
	bsHasAVX2 = false
	defer func() { bsHasAVX2 = true }()
	TestThreeBandFilterBankKernel(t)
	TestThreeBandFilterBankSignedZeros(t)
	TestAudioUtilKernels(t)
}
