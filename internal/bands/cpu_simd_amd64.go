//go:build goexperiment.simd && amd64 && !purego

package bands

import "simd/archsimd"

// bsHasSIMD reports whether the archsimd kernels can run: the 128-bit
// Float32x4 operations are VEX-encoded and need AVX.
var bsHasSIMD = archsimd.X86.AVX()
