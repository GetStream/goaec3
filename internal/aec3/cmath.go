package aec3

import "math"

// Float transcendentals used on the AEC3 signal path. libwebrtc calls the
// platform libm (clang lowers std::pow(2.0f, x) to exp2f and
// powf(10.f, x) to __exp10f), whose float results are not correctly rounded
// and differ between Apple libm and glibc. The port pins them to the
// correctly rounded value of the float64 function; the C++ oracle binary
// interposes the same Go functions for exp2f/__exp10f/log10f so parity tests compare
// the AEC3 algorithm, not the host libm.

// Exp2f returns 2**x rounded to float32.
func Exp2f(x float32) float32 { return float32(math.Exp2(float64(x))) }

// Log10f returns log10(x) rounded to float32 (used by the metrics path; the
// host log10f is not correctly rounded, e.g. log10f(1.1627032f) is 1 ulp off).
func Log10f(x float32) float32 { return float32(math.Log10(float64(x))) }

// Exp10f returns 10**x rounded to float32.
func Exp10f(x float32) float32 { return float32(math.Pow(10, float64(x))) }
