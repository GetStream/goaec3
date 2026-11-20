//go:build goexperiment.simd && amd64 && !purego

package bands

import "simd/archsimd"

// bsHasSIMD reports whether the archsimd kernels can run: the 128-bit
// Float32x4 operations are VEX-encoded and need AVX.
var bsHasSIMD = archsimd.X86.AVX()

// bsSplat4 returns x in all four lanes with AVX instructions (VMOVSS,
// VSHUFPS). archsimd.BroadcastFloat32x4 compiles to VBROADCASTSS from a
// register, and BroadcastUint32x4 to VPBROADCASTD, which are AVX2
// instructions: the 128-bit kernels run on CPUs with AVX only.
func bsSplat4(x float32) archsimd.Float32x4 {
	var z archsimd.Float32x4
	v := z.SetElem(0, x)
	return v.ConcatPermuteScalars(0, 0, 0, 0, v)
}

// bsSplatU4 is bsSplat4 for uint32 lanes (VMOVD, VPSHUFD).
func bsSplatU4(x uint32) archsimd.Uint32x4 {
	var z archsimd.Uint32x4
	return z.SetElem(0, x).PermuteScalars(0, 0, 0, 0)
}
