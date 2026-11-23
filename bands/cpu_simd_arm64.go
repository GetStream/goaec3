//go:build goexperiment.simd && arm64 && !purego

package bands

import "simd/archsimd"

// bsHasSIMD reports whether the archsimd kernels can run: NEON is always
// present on arm64.
const bsHasSIMD = true

// bsSplat4 returns x in all four lanes.
func bsSplat4(x float32) archsimd.Float32x4 { return archsimd.BroadcastFloat32x4(x) }

// bsSplatU4 is bsSplat4 for uint32 lanes.
func bsSplatU4(x uint32) archsimd.Uint32x4 { return archsimd.BroadcastUint32x4(x) }
