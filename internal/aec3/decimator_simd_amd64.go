//go:build goexperiment.simd && amd64 && !purego

package aec3

// dlyDs4HasSimd reports that the CPU runs dlyDecimateDs4, whose 128-bit
// operations are VEX encoded.
var dlyDs4HasSimd = kernHasAVX
