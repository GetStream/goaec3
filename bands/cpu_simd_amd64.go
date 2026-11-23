//go:build goexperiment.simd && amd64 && !purego

package bands

import "simd/archsimd"

// bsHasSIMD reports whether the archsimd kernels can run: the 128-bit
// Float32x4 operations are VEX-encoded and need AVX.
var bsHasSIMD = archsimd.X86.AVX()

// bsSplat4 returns x in all four lanes using AVX instructions only.
// archsimd.BroadcastFloat32x4 and BroadcastUint32x4 compile to AVX2
// instructions, and the 128-bit kernels run on CPUs without AVX2.
func bsSplat4(x float32) archsimd.Float32x4 {
	var z archsimd.Float32x4
	v := z.SetElem(0, x)
	return v.ConcatPermuteScalars(0, 0, 0, 0, v)
}

// bsSplatU4 is bsSplat4 for uint32 lanes.
func bsSplatU4(x uint32) archsimd.Uint32x4 {
	var z archsimd.Uint32x4
	return z.SetElem(0, x).PermuteScalars(0, 0, 0, 0)
}
