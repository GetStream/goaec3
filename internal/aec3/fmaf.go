package aec3

import "math"

// fmaf returns a*b + c rounded once to float32: C fmaf, and a lane of
// vfmaq_f32 and _mm256_fmadd_ps.
//
// The product p = a*b is exact in float64: it has at most 48 significant
// bits and a magnitude in [2^-298, 2^256). So s = fl64(p + c) is one
// rounding of the exact sum x, and it neither overflows nor underflows.
// float32(s) is the result unless s is a float32 rounding boundary: a
// midpoint of two adjacent float32 values or ±(2^128 - 2^103). The
// boundaries are float64 values, and rounding to float64 is monotone, so x
// and s lie between the same two boundaries unless s is one, and round to
// the same float32.
//
// A sum that is a boundary takes fmafSlow (fmafHard, fmafTie). So does every
// sum with 0 < |s| < 2^-126, where the test of the low bits does not find
// the boundaries. A zero s is an exact zero with the sign of the fused
// result; Inf and NaN convert as they are.
//
// fmaf does not inline. A loop that calls it spills what it keeps in
// registers on every iteration; loops compute fmafSum for a group of
// elements, test the sums in one condition and, on a hit, leave for a
// //go:noinline function that calls fmaf per element.
func fmaf(a, b, c float32) float32 {
	s := fmafSum(a, b, c)
	if fmafHard(s) {
		return fmafSlow(a, b, c)
	}
	return float32(s)
}

// fmafSum returns fl64(a*b + c). The conversion of the product keeps the
// compiler from fusing it into the sum, which would give the same value.
func fmafSum(a, b, c float32) float64 {
	return float64(float64(a)*float64(b)) + float64(c)
}

// fmafHard reports whether float32(s) may differ from fmaf(a, b, c) for s =
// fmafSum(a, b, c): the low 29 bits of s are 1000...0, as in a boundary at a
// float32 normal exponent or above, or 0 < |s| < 2^-126. The subtraction
// wraps for ±0, which is not hard.
func fmafHard(s float64) bool {
	bits := math.Float64bits(s)
	return bits&0x1FFFFFFF == 0x10000000 || bits<<1-1 < 897<<53-1
}

// fmafTie is fmafHard for a loop that has r = float32(s) and t = float64(r)
// anyway. It holds for every float32 rounding boundary:
//
// A boundary has at most 25 significant bits and a magnitude of at least
// 2^-150, so it is a normal float64 whose 28 low significand bits are zero.
// And it is not a float32 value, so t != s, also when r is Inf.
//
// A sum that is a float32 value has t == s, and a NaN has s != s: neither is
// a tie.
//
// math.FMA on the widened operands returns the same s, since the float64
// product is exact, and needs the same test.
func fmafTie(s, t float64) bool {
	return math.Float64bits(s)&fmafLowBits == 0 && t != s && s == s
}

// fmafLowBits holds the float64 significand bits that are zero in every
// float32 rounding boundary.
const fmafLowBits = 0x0FFFFFFF

// fmafSlow is fmaf without the fast path. It rounds the float64 sum to odd,
// using the exact error term of TwoSum, which makes the rounding to float32
// correct: 53 >= 24+2 significand bits.
//
//go:noinline
func fmafSlow(a, b, c float32) float32 {
	p := float64(float64(a) * float64(b))
	q := float64(c)
	s := p + q
	if math.IsInf(s, 0) || math.IsNaN(s) || s == 0 {
		// float32 operands cannot overflow. Inf and NaN convert as they are,
		// and an exact zero has the sign of IEEE addition.
		return float32(s)
	}
	// TwoSum: s + e == p + q exactly.
	bv := s - p
	e := (p - (s - bv)) + (q - bv)
	if e != 0 {
		bits := math.Float64bits(s)
		if bits&1 == 0 {
			// Round to odd: move s one ulp toward the exact value.
			if (e > 0) == (s > 0) {
				bits++
			} else {
				bits--
			}
			s = math.Float64frombits(bits)
		}
	}
	return float32(s)
}
