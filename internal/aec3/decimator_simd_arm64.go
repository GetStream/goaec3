//go:build goexperiment.simd && arm64 && !purego

package aec3

// dlyDs4HasSimd reports whether the CPU runs dlyDecimateDs4.
const dlyDs4HasSimd = true
