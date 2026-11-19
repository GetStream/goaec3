//go:build goexperiment.simd && amd64 && !purego

package aec3

import (
	"os"
	"simd/archsimd"
	"testing"
)

func TestDlyAmd64KernelSelection(t *testing.T) {
	t.Logf("AVX=%v AVX2=%v FMA=%v", archsimd.X86.AVX(), archsimd.X86.AVX2(), archsimd.X86.FMA())
	t.Logf("SSE2-lane archsimd kernels: %v, AVX2-lane archsimd kernels: %v", dlyUseAvx, dlyUseFma)
	if !dlyUseFma {
		t.Log("AVX2/FMA not available: the AVX2 lane runs its pure-Go reference")
	}
}

// TestDlyAvx2KernelsForced runs the AVX2-lane archsimd kernels against their
// references even when CPUID does not report FMA (set DLY_FORCE_FMA=1; some
// emulators execute FMA instructions without advertising them).
func TestDlyAvx2KernelsForced(t *testing.T) {
	if os.Getenv("DLY_FORCE_FMA") == "" {
		t.Skip("set DLY_FORCE_FMA=1 to force the AVX2-lane kernels")
	}
	saved := dlyUseFma
	dlyUseFma = true
	defer func() { dlyUseFma = saved }()
	TestDlyMatchedFilterKernelsMatchReference(t)
	TestDlyMatchedFilterCoreLanesMatchReference(t)
	TestDlyMatchedFilterAccErrCoresMatchReference(t)
}
