package aec3

import "math"

// fmaf returns a*b+c rounded once to float32 (round to nearest, ties to
// even), the result of C fmaf, vfmaq_f32 and _mm256_fmadd_ps per lane. It is
// the software fused multiply-add of the AVX2 lane references.
//
// Fast path. The product p = a*b of two float32 values is exact in float64:
// it has at most 48 significant bits and |p| lies in [2^-298, 2^256), inside
// the float64 normal range. Let x = p + c be the exact sum and s = fl64(x) its
// float64 rounding (one float64 addition; it cannot overflow, and since p and
// c are multiples of 2^-298 it cannot underflow either). The wanted result is
// r32(x), where r32 rounds to float32 and handles overflow to ±Inf. Then
// float32(s) = r32(s), and r32(s) = r32(x) whenever s is not a float32
// rounding boundary:
//
//   - Let M be the set of float32 rounding boundaries: the midpoints between
//     adjacent float32 values (including the subnormal ones) and ±(2^128 -
//     2^103), above which r32 overflows. Each m in M has at most 25
//     significant bits and exponent >= -150, so m is a float64 value.
//   - r32 is constant on each open interval between adjacent boundaries.
//   - fl64 is monotone and fixes float64 values, so for boundaries m1 < m2
//     with m1 < x < m2, m1 <= s <= m2. If s is not itself a boundary, s and
//     x lie in the same open interval and r32(s) = r32(x). If x is a
//     boundary then x = s exactly and float32(s) is correct as well, but
//     s being a boundary does not tell the two cases apart, so the slow path
//     handles every s in M.
//
// Detecting s in M. For |s| >= 2^-126 (biased float64 exponent >= 897), the
// float32 boundaries at that exponent are exactly the float64 values whose
// 29 low significand bits (those below the float32 significand) are
// 1000...0, i.e. bits&0x1FFFFFFF == 0x10000000; this includes 2^128 - 2^103
// (exponent 127, 24 one bits below the leading bit). For larger exponents the
// test only sends values to the slow path that need not go there. For
// 0 < |s| < 2^-126 the boundaries have fewer significand bits, and the fast
// path is not taken at all. s = ±0 means x = 0 exactly; IEEE addition then
// gives the same signed zero as the fused operation. Inf and NaN operands
// give an Inf or NaN s (exponent 2047, low bits of Inf are zero) and float32
// conversion propagates them like the hardware; a NaN whose low bits match
// the pattern goes to the slow path, which also propagates it.
//
// The product is wrapped in float64(...) so the compiler does not contract
// it into the add; contraction would compute the same s (the product is
// exact), but the explicit rounding keeps the fused-op guard meaningful.
//
// fmaf does not inline (the slow-path call exceeds the inlining budget). Hot
// loops expand it with the inlinable fmafSum and fmafHard instead:
//
//	s := fmafSum(a, b, c)
//	r := float32(s)
//	if fmafHard(s) {
//		r = fmafSlow(a, b, c)
//	}
//
// A call inside a loop makes the compiler keep everything that lives across
// it (the counter, pointers, accumulators) in stack slots that the loop
// writes on every iteration, taken or not. The form that avoids it keeps the
// loop free of calls: compute fmafSum for a group of elements, test them with
// one `if fmafHard(s0) || fmafHard(s1) || ...`, and on a hit hand the rest of
// the work to a //go:noinline function that uses fmaf, then return
// (kernSpectrumAvx2Ref in kern_ref.go). Nothing is live after that call, so
// the loop spills nothing.
func fmaf(a, b, c float32) float32 {
	s := fmafSum(a, b, c)
	if fmafHard(s) {
		return fmafSlow(a, b, c)
	}
	return float32(s)
}

// fmafSum returns s = fl64(a*b + c), the float64 sum of fmaf's fast path.
func fmafSum(a, b, c float32) float64 {
	return float64(float64(a)*float64(b)) + float64(c)
}

// fmafHard reports whether float32(s) may differ from fmaf(a, b, c) for
// s = fmafSum(a, b, c): s is a float32 rounding boundary at a float32 normal
// exponent or above (low 29 bits 1000...0), or 0 < |s| < 2^-126. bits<<1
// drops the sign, and the subtraction wraps for ±0, which stays on the fast
// path.
func fmafHard(s float64) bool {
	bits := math.Float64bits(s)
	return bits&0x1FFFFFFF == 0x10000000 || bits<<1-1 < 897<<53-1
}

// fmafSlow is fmaf without the fast path. It computes the float64 sum with
// an exact error term (TwoSum) and rounds it to odd, which makes the final
// rounding to float32 correct (53 >= 24+2 significand bits, also for float32
// subnormal results, which have fewer bits).
//
//go:noinline
func fmafSlow(a, b, c float32) float32 {
	p := float64(float64(a) * float64(b))
	q := float64(c)
	s := p + q
	if math.IsInf(s, 0) || math.IsNaN(s) || s == 0 {
		// Overflow cannot happen for float32 operands; Inf/NaN propagate, and
		// an exact zero needs no rounding (its sign follows IEEE addition).
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
