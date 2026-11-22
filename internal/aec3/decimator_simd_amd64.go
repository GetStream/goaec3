//go:build goexperiment.simd && amd64 && !purego

package aec3

// dlyDs4HasSimd reports whether the CPU runs dlyDecimateDs4, which is VEX
// encoded.
var dlyDs4HasSimd = kernHasAVX
