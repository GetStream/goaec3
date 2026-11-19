//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// archsimd (NEON) kernels for the kern_bins.go helpers.

func kernScale(x []float32, a float32, z []float32) {
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] * a
		}
		kernScale64((*[64]float32)(x), a, (*[64]float32)(z))
		return
	}
	av := archsimd.BroadcastFloat32x4(a)
	j := 0
	for ; j+8 <= len(x); j += 8 {
		archsimd.LoadFloat32x4(x[j:]).Mul(av).Store(z[j:])
		archsimd.LoadFloat32x4(x[j+4:]).Mul(av).Store(z[j+4:])
	}
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(x[j:]).Mul(av).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = x[j] * a
	}
}

func kernAdd(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] + y[64]
		}
		kernAdd64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		return
	}
	j := 0
	for ; j+8 <= len(x); j += 8 {
		archsimd.LoadFloat32x4(x[j:]).Add(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
		archsimd.LoadFloat32x4(x[j+4:]).Add(archsimd.LoadFloat32x4(y[j+4:])).Store(z[j+4:])
	}
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(x[j:]).Add(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = x[j] + y[j]
	}
}

func kernSub(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] - y[64]
		}
		kernSub64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		return
	}
	j := 0
	for ; j+8 <= len(x); j += 8 {
		archsimd.LoadFloat32x4(x[j:]).Sub(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
		archsimd.LoadFloat32x4(x[j+4:]).Sub(archsimd.LoadFloat32x4(y[j+4:])).Store(z[j+4:])
	}
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(x[j:]).Sub(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = x[j] - y[j]
	}
}

// kernSumRows keeps sixteen bins of dst in registers while all rows
// accumulate into them, in row order starting from +0.
func kernSumRows(dst *[FftLengthBy2Plus1]float32, rows []*[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k += 16 {
		var a0, a1, a2, a3 f32x4
		for _, r := range rows {
			p := (*[16]float32)(r[k : k+16])
			a0 = a0.Add(archsimd.LoadFloat32x4Array((*[4]float32)(p[0:4])))
			a1 = a1.Add(archsimd.LoadFloat32x4Array((*[4]float32)(p[4:8])))
			a2 = a2.Add(archsimd.LoadFloat32x4Array((*[4]float32)(p[8:12])))
			a3 = a3.Add(archsimd.LoadFloat32x4Array((*[4]float32)(p[12:16])))
		}
		p := (*[16]float32)(dst[k : k+16])
		a0.StoreArray((*[4]float32)(p[0:4]))
		a1.StoreArray((*[4]float32)(p[4:8]))
		a2.StoreArray((*[4]float32)(p[8:12]))
		a3.StoreArray((*[4]float32)(p[12:16]))
	}
	var last float32
	for _, r := range rows {
		last += r[FftLengthBy2]
	}
	dst[FftLengthBy2] = last
}

// kernUnpack deinterleaves v into re[0..63] (even) and im[0..63] (odd),
// then moves the Nyquist real part from im[0] to re[64].
func kernUnpack(v *[FftLength]float32, re, im *[FftLengthBy2Plus1]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		p := (*[8]float32)(v[2*k : 2*k+8])
		a := archsimd.LoadFloat32x4Array((*[4]float32)(p[0:4])).ToBits()
		b := archsimd.LoadFloat32x4Array((*[4]float32)(p[4:8])).ToBits()
		a.ConcatEven(b).BitsToFloat32().StoreArray((*[4]float32)(re[k : k+4]))
		a.ConcatOdd(b).BitsToFloat32().StoreArray((*[4]float32)(im[k : k+4]))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
	re[FftLengthBy2] = im[0]
	im[0] = 0
	im[FftLengthBy2] = 0
}

// kernPack interleaves re[0..63] and im[0..63] into v, then puts the Nyquist
// real part re[64] in v[1].
func kernPack(re, im *[FftLengthBy2Plus1]float32, v *[FftLength]float32) {
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		a := archsimd.LoadFloat32x4Array((*[4]float32)(re[k : k+4])).ToBits()
		b := archsimd.LoadFloat32x4Array((*[4]float32)(im[k : k+4])).ToBits()
		p := (*[8]float32)(v[2*k : 2*k+8])
		a.InterleaveLo(b).BitsToFloat32().StoreArray((*[4]float32)(p[0:4]))
		a.InterleaveHi(b).BitsToFloat32().StoreArray((*[4]float32)(p[4:8]))
	}
	g := func(k int) {
		_ = zero
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
	v[1] = re[FftLengthBy2]
}

// kernMinMaxFast reports whether kernMinMax has a fast path.
const kernMinMaxFast = true

// kernMinMax returns the minimum and maximum of x with ok = true when they
// are exact for every compare-loop formulation (first or last occurrence,
// strict or non-strict comparisons, std::min_element/std::max_element and
// the pairwise std::minmax_element): FMIN/FMAX give the same extreme values
// as such loops unless a NaN is present (FMIN/FMAX propagate it, the loops
// skip or pick it by position) or an extreme is zero (the loops pick -0 or
// +0 by position). Those cases return ok = false and the caller runs its
// own loop. Any other extreme value has a single bit pattern.
func kernMinMax(x []float32) (mn, mx float32, ok bool) {
	if len(x) >= 8 {
		var vmn, vmx f32x4
		j := 64
		if len(x) == 64 {
			vmn, vmx = kernMinMax64((*[64]float32)(x))
		} else {
			vmn = archsimd.LoadFloat32x4(x)
			vmx = vmn
			for j = 4; j+4 <= len(x); j += 4 {
				v := archsimd.LoadFloat32x4(x[j:])
				vmn = vmn.Min(v)
				vmx = vmx.Max(v)
			}
		}
		mn, mx = vmn.ReduceMin(), vmx.ReduceMax()
		for ; j < len(x); j++ {
			mn = min(mn, x[j])
			mx = max(mx, x[j])
		}
		if mn == mn && mx == mx && mn != 0 && mx != 0 {
			return mn, mx, true
		}
	}
	return 0, 0, false
}

// kernClamp is kernClampGeneric: std::min(std::max(x, lo), hi).
func kernClamp(x []float32, lo, hi float32) {
	if kernIs64(len(x)) {
		if len(x) == 65 {
			x[64] = stdMin(stdMax(x[64], lo), hi)
		}
		kernClamp64((*[64]float32)(x), lo, hi)
		return
	}
	lv, hv := archsimd.BroadcastFloat32x4(lo), archsimd.BroadcastFloat32x4(hi)
	j := 0
	for ; j+8 <= len(x); j += 8 {
		vStdMin(vStdMax(archsimd.LoadFloat32x4(x[j:]), lv), hv).Store(x[j:])
		vStdMin(vStdMax(archsimd.LoadFloat32x4(x[j+4:]), lv), hv).Store(x[j+4:])
	}
	for ; j+4 <= len(x); j += 4 {
		vStdMin(vStdMax(archsimd.LoadFloat32x4(x[j:]), lv), hv).Store(x[j:])
	}
	for ; j < len(x); j++ {
		x[j] = stdMin(stdMax(x[j], lo), hi)
	}
}

// kernHas128 reports whether the 128-bit kernels run (always on arm64).
const kernHas128 = true

// kernMinMaxStep folds x[k:k+4] into the lane minima and maxima.
func kernMinMaxStep(x *[64]float32, k int, mn, mx f32x4) (f32x4, f32x4) {
	v := kernLd64(x, k)
	return mn.Min(v), mx.Max(v)
}

// kernMinMax64 returns the lane minima and maxima of a block. FMIN and FMAX
// propagate NaN and order -0 below +0 whatever the operand order, so the
// block is folded as four independent chains.
func kernMinMax64(x *[64]float32) (f32x4, f32x4) {
	mn0, mn1, mn2, mn3 := kernLd64(x, 0), kernLd64(x, 4), kernLd64(x, 8), kernLd64(x, 12)
	mx0, mx1, mx2, mx3 := mn0, mn1, mn2, mn3
	mn0, mx0 = kernMinMaxStep(x, 16, mn0, mx0)
	mn1, mx1 = kernMinMaxStep(x, 20, mn1, mx1)
	mn2, mx2 = kernMinMaxStep(x, 24, mn2, mx2)
	mn3, mx3 = kernMinMaxStep(x, 28, mn3, mx3)
	mn0, mx0 = kernMinMaxStep(x, 32, mn0, mx0)
	mn1, mx1 = kernMinMaxStep(x, 36, mn1, mx1)
	mn2, mx2 = kernMinMaxStep(x, 40, mn2, mx2)
	mn3, mx3 = kernMinMaxStep(x, 44, mn3, mx3)
	mn0, mx0 = kernMinMaxStep(x, 48, mn0, mx0)
	mn1, mx1 = kernMinMaxStep(x, 52, mn1, mx1)
	mn2, mx2 = kernMinMaxStep(x, 56, mn2, mx2)
	mn3, mx3 = kernMinMaxStep(x, 60, mn3, mx3)
	return mn0.Min(mn1).Min(mn2.Min(mn3)), mx0.Max(mx1).Max(mx2.Max(mx3))
}
