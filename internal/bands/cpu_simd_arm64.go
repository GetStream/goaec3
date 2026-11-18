//go:build goexperiment.simd && arm64 && !purego

package bands

// bsHasSIMD reports whether the archsimd kernels can run: NEON is always
// present on arm64.
const bsHasSIMD = true
