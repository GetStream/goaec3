//go:build goexperiment.simd && arm64 && !purego

package aec3

// dlyDs4HasSimd is true: every arm64 CPU has NEON.
const dlyDs4HasSimd = true
