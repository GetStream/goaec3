package aec3

import "math"

// Float transcendentals of the signal path. libwebrtc calls the host libm
// (clang lowers std::pow(2.0f, x) to exp2f and powf(10.f, x) to __exp10f),
// whose float results are not correctly rounded and differ between
// platforms. The port uses the rounded result of the float64 function, and
// the oracle binary interposes these functions for the libm symbols.

// Exp2f returns 2**x rounded to float32.
func Exp2f(x float32) float32 { return float32(math.Exp2(float64(x))) }

// Log10f returns log10(x) rounded to float32.
func Log10f(x float32) float32 { return float32(math.Log10(float64(x))) }

// Exp10f returns 10**x rounded to float32.
func Exp10f(x float32) float32 { return float32(math.Pow(10, float64(x))) }
