//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// archsimd kernels for the kern_bins.go helpers: 256-bit code with AVX2,
// 128-bit (VEX) code with AVX, the references otherwise.
//
// Every 256-bit section ends with archsimd.ClearAVXUpperBits (VZEROUPPER).
// Go's scalar float code is legacy-SSE encoded, and on Intel cores legacy
// SSE instructions issued while the upper halves of the YMM registers are
// dirty run with a large penalty (false dependencies and state
// transitions), which slows every scalar float operation that follows.

func kernScale(x []float32, a float32, z []float32) {
	z = z[:len(x)]
	if kernHasAVX && kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] * a
		}
		if kernHasAVX2 {
			kernScale64y((*[64]float32)(x), a, (*[64]float32)(z))
		} else {
			kernScale64((*[64]float32)(x), a, (*[64]float32)(z))
		}
		return
	}
	j := 0
	if kernHasAVX2 {
		j = kernScale256(x, a, z)
	}
	if kernHasAVX {
		av := archsimd.BroadcastFloat32x4(a)
		for ; j+4 <= len(x); j += 4 {
			archsimd.LoadFloat32x4(x[j:]).Mul(av).Store(z[j:])
		}
	}
	for ; j < len(x); j++ {
		z[j] = x[j] * a
	}
}

func kernAdd(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	if kernHasAVX && kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] + y[64]
		}
		if kernHasAVX2 {
			kernAdd64y((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		} else {
			kernAdd64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		}
		return
	}
	j := 0
	if kernHasAVX2 {
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(x[j:]).Add(archsimd.LoadFloat32x8(y[j:])).Store(z[j:])
		}
		archsimd.ClearAVXUpperBits()
	}
	if kernHasAVX {
		for ; j+4 <= len(x); j += 4 {
			archsimd.LoadFloat32x4(x[j:]).Add(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
		}
	}
	for ; j < len(x); j++ {
		z[j] = x[j] + y[j]
	}
}

func kernSub(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	if kernHasAVX && kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] - y[64]
		}
		if kernHasAVX2 {
			kernSub64y((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		} else {
			kernSub64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		}
		return
	}
	j := 0
	if kernHasAVX2 {
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(x[j:]).Sub(archsimd.LoadFloat32x8(y[j:])).Store(z[j:])
		}
		archsimd.ClearAVXUpperBits()
	}
	if kernHasAVX {
		for ; j+4 <= len(x); j += 4 {
			archsimd.LoadFloat32x4(x[j:]).Sub(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
		}
	}
	for ; j < len(x); j++ {
		z[j] = x[j] - y[j]
	}
}

// kernSumRows keeps sixteen bins of dst in registers while all rows
// accumulate into them, in row order starting from +0.
func kernSumRows(dst *[FftLengthBy2Plus1]float32, rows []*[FftLengthBy2Plus1]float32) {
	if !kernHasAVX2 {
		kernSumRows128(dst, rows)
		return
	}
	for k := 0; k < FftLengthBy2; k += 16 {
		var a0, a1 archsimd.Float32x8
		for _, r := range rows {
			p := (*[16]float32)(r[k : k+16])
			a0 = a0.Add(archsimd.LoadFloat32x8Array((*[8]float32)(p[0:8])))
			a1 = a1.Add(archsimd.LoadFloat32x8Array((*[8]float32)(p[8:16])))
		}
		p := (*[16]float32)(dst[k : k+16])
		a0.StoreArray((*[8]float32)(p[0:8]))
		a1.StoreArray((*[8]float32)(p[8:16]))
	}
	archsimd.ClearAVXUpperBits()
	var last float32
	for _, r := range rows {
		last += r[FftLengthBy2]
	}
	dst[FftLengthBy2] = last
}

// kernSumRows128 is kernSumRows with 128-bit operations (AVX without AVX2).
func kernSumRows128(dst *[FftLengthBy2Plus1]float32, rows []*[FftLengthBy2Plus1]float32) {
	if !kernHasAVX {
		kernSumRowsGeneric(dst, rows)
		return
	}
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
	if !kernHasAVX {
		kernUnpackGeneric(v, re, im)
		return
	}
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		p := (*[8]float32)(v[2*k : 2*k+8])
		a := archsimd.LoadFloat32x4Array((*[4]float32)(p[0:4]))
		b := archsimd.LoadFloat32x4Array((*[4]float32)(p[4:8]))
		a.ConcatPermuteScalars(0, 2, 4, 6, b).StoreArray((*[4]float32)(re[k : k+4]))
		a.ConcatPermuteScalars(1, 3, 5, 7, b).StoreArray((*[4]float32)(im[k : k+4]))
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
	if !kernHasAVX {
		kernPackGeneric(re, im, v)
		return
	}
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

// kernHas128 reports whether the 128-bit kernels of kern_bins_simd128.go
// run; their VEX encoding needs AVX.
var kernHas128 = archsimd.X86.AVX()

// kernMinMaxFast reports whether kernMinMax has a fast path.
const kernMinMaxFast = true

// kernMinMax returns the minimum and maximum of x with ok = true when they
// are exact for every compare-loop formulation (see the arm64 kernMinMax):
// no element is NaN and neither extreme is zero, so each extreme has a
// single bit pattern. MINPS/MAXPS drop a NaN operand instead of propagating
// it, so NaN is detected by its own unordered compare.
func kernMinMax(x []float32) (mn, mx float32, ok bool) {
	if !kernHas128 || len(x) < 8 {
		return 0, 0, false
	}
	if len(x) == 64 {
		return kernMinMax64((*[64]float32)(x))
	}
	vmn := archsimd.LoadFloat32x4(x)
	vmx := vmn
	nan := vmn.IsNaN()
	j := 4
	for ; j+4 <= len(x); j += 4 {
		v := archsimd.LoadFloat32x4(x[j:])
		vmn = vmn.Min(v)
		vmx = vmx.Max(v)
		nan = nan.Or(v.IsNaN())
	}
	if nan.ToBits() != 0 {
		return 0, 0, false
	}
	var a, b [4]float32
	vmn.StoreArray(&a)
	vmx.StoreArray(&b)
	mn, mx = min(a[0], a[1], a[2], a[3]), max(b[0], b[1], b[2], b[3])
	for ; j < len(x); j++ {
		mn = min(mn, x[j])
		mx = max(mx, x[j])
	}
	if mn == mn && mx == mx && mn != 0 && mx != 0 {
		return mn, mx, true
	}
	return 0, 0, false
}

// kernClamp is kernClampGeneric, std::min(std::max(x, lo), hi), as compares
// and selects: MINPS/MAXPS return their second operand on NaN and equal
// zeros, and archsimd does not fix which operand that is.
func kernClamp(x []float32, lo, hi float32) {
	if kernHasAVX && kernIs64(len(x)) {
		if len(x) == 65 {
			x[64] = stdMin(stdMax(x[64], lo), hi)
		}
		if kernHasAVX2 {
			kernClamp64y((*[64]float32)(x), lo, hi)
		} else {
			kernClamp64((*[64]float32)(x), lo, hi)
		}
		return
	}
	j := 0
	if kernHasAVX2 {
		j = kernClamp256(x, lo, hi)
	}
	if kernHasAVX {
		lv, hv := archsimd.BroadcastFloat32x4(lo), archsimd.BroadcastFloat32x4(hi)
		for ; j+4 <= len(x); j += 4 {
			v := archsimd.LoadFloat32x4(x[j:])
			v = lv.IfElse(v.Less(lv), v)
			hv.IfElse(hv.Less(v), v).Store(x[j:])
		}
	}
	for ; j < len(x); j++ {
		x[j] = stdMin(stdMax(x[j], lo), hi)
	}
}

// kernScale256 and kernClamp256 are the 256-bit loops of kernScale and
// kernClamp; they return the number of elements done. They do not inline:
// a float32 live across ClearAVXUpperBits is spilled with a 4-byte store
// and restored with a 16-byte load, which the CPU cannot forward from that
// store, and the callers' scalar tails use a, lo and hi.
//
//go:noinline
func kernScale256(x []float32, a float32, z []float32) int {
	z = z[:len(x)]
	av := archsimd.BroadcastFloat32x8(a)
	j := 0
	for ; j+8 <= len(x); j += 8 {
		archsimd.LoadFloat32x8(x[j:]).Mul(av).Store(z[j:])
	}
	archsimd.ClearAVXUpperBits()
	return j
}

//go:noinline
func kernClamp256(x []float32, lo, hi float32) int {
	lv, hv := archsimd.BroadcastFloat32x8(lo), archsimd.BroadcastFloat32x8(hi)
	j := 0
	for ; j+8 <= len(x); j += 8 {
		v := archsimd.LoadFloat32x8(x[j:])
		v = lv.IfElse(v.Less(lv), v)
		hv.IfElse(hv.Less(v), v).Store(x[j:])
	}
	archsimd.ClearAVXUpperBits()
	return j
}

// 256-bit forms of the 64-element kernels in kern_bins_simd64.go: eight
// steps with constant offsets, then VZEROUPPER. Nothing is live after it,
// and the helpers store the 65th element before the call.

func kernLd64y(a *[64]float32, k int) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[8]float32)(a[k : k+8]))
}

func kernSt64y(a *[64]float32, k int, v archsimd.Float32x8) {
	v.StoreArray((*[8]float32)(a[k : k+8]))
}

func kernScale64y(x *[64]float32, a float32, z *[64]float32) {
	av := archsimd.BroadcastFloat32x8(a)
	f := func(k int) {
		_ = av // a captured vector makes the compiler inline the closure
		kernSt64y(z, k, kernLd64y(x, k).Mul(av))
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
}

func kernAdd64y(x, y, z *[64]float32) {
	var zero archsimd.Float32x8
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64y(z, k, kernLd64y(x, k).Add(kernLd64y(y, k)))
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
}

func kernSub64y(x, y, z *[64]float32) {
	var zero archsimd.Float32x8
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64y(z, k, kernLd64y(x, k).Sub(kernLd64y(y, k)))
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
}

func kernMul64y(x, y, z *[64]float32) {
	var zero archsimd.Float32x8
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64y(z, k, kernLd64y(x, k).Mul(kernLd64y(y, k)))
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
}

func kernSqrt64y(x *[64]float32) {
	var zero archsimd.Float32x8
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		kernSt64y(x, k, kernLd64y(x, k).Sqrt())
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
}

func kernClamp64y(x *[64]float32, lo, hi float32) {
	lv, hv := archsimd.BroadcastFloat32x8(lo), archsimd.BroadcastFloat32x8(hi)
	f := func(k int) {
		_ = lv // a captured vector makes the compiler inline the closure
		v := kernLd64y(x, k)
		v = lv.IfElse(v.Less(lv), v)
		kernSt64y(x, k, hv.IfElse(hv.Less(v), v))
	}
	f(0)
	f(8)
	f(16)
	f(24)
	f(32)
	f(40)
	f(48)
	f(56)
	archsimd.ClearAVXUpperBits()
}

// kernMinMax64 is kernMinMax for a block. MINPS and MAXPS are order
// dependent only on NaN and on equal zeros; a NaN is detected by its own
// compare and a zero extreme is rejected, so the block is folded as
// independent chains.
func kernMinMax64(x *[64]float32) (mn, mx float32, ok bool) {
	var a, b [4]float32
	var nan bool
	if kernHasAVX2 {
		nan = kernMinMax64y(x, &a, &b)
	} else {
		nan = kernMinMax64x(x, &a, &b)
	}
	if nan {
		return 0, 0, false
	}
	mn, mx = min(a[0], a[1], a[2], a[3]), max(b[0], b[1], b[2], b[3])
	if mn == mn && mx == mx && mn != 0 && mx != 0 {
		return mn, mx, true
	}
	return 0, 0, false
}

func kernMinMaxStep(x *[64]float32, k int, mn, mx f32x4, nan archsimd.Mask32x4) (f32x4, f32x4, archsimd.Mask32x4) {
	v := kernLd64(x, k)
	return mn.Min(v), mx.Max(v), nan.Or(v.IsNaN())
}

// kernMinMax64x stores the lane minima and maxima of x in a and b and
// reports whether x holds a NaN.
func kernMinMax64x(x *[64]float32, a, b *[4]float32) bool {
	mn0, mn1, mn2, mn3 := kernLd64(x, 0), kernLd64(x, 4), kernLd64(x, 8), kernLd64(x, 12)
	mx0, mx1, mx2, mx3 := mn0, mn1, mn2, mn3
	n0, n1, n2, n3 := mn0.IsNaN(), mn1.IsNaN(), mn2.IsNaN(), mn3.IsNaN()
	mn0, mx0, n0 = kernMinMaxStep(x, 16, mn0, mx0, n0)
	mn1, mx1, n1 = kernMinMaxStep(x, 20, mn1, mx1, n1)
	mn2, mx2, n2 = kernMinMaxStep(x, 24, mn2, mx2, n2)
	mn3, mx3, n3 = kernMinMaxStep(x, 28, mn3, mx3, n3)
	mn0, mx0, n0 = kernMinMaxStep(x, 32, mn0, mx0, n0)
	mn1, mx1, n1 = kernMinMaxStep(x, 36, mn1, mx1, n1)
	mn2, mx2, n2 = kernMinMaxStep(x, 40, mn2, mx2, n2)
	mn3, mx3, n3 = kernMinMaxStep(x, 44, mn3, mx3, n3)
	mn0, mx0, n0 = kernMinMaxStep(x, 48, mn0, mx0, n0)
	mn1, mx1, n1 = kernMinMaxStep(x, 52, mn1, mx1, n1)
	mn2, mx2, n2 = kernMinMaxStep(x, 56, mn2, mx2, n2)
	mn3, mx3, n3 = kernMinMaxStep(x, 60, mn3, mx3, n3)
	mn0.Min(mn1).Min(mn2.Min(mn3)).StoreArray(a)
	mx0.Max(mx1).Max(mx2.Max(mx3)).StoreArray(b)
	return n0.Or(n1).Or(n2.Or(n3)).ToBits() != 0
}

func kernMinMaxStepY(x *[64]float32, k int, mn, mx archsimd.Float32x8, nan archsimd.Mask32x8) (archsimd.Float32x8, archsimd.Float32x8, archsimd.Mask32x8) {
	v := kernLd64y(x, k)
	return mn.Min(v), mx.Max(v), nan.Or(v.IsNaN())
}

// kernMinMax64y is kernMinMax64x with 256-bit operations. The results leave
// through a and b, which the caller reads after VZEROUPPER.
//
//go:noinline
func kernMinMax64y(x *[64]float32, a, b *[4]float32) bool {
	mn0, mn1 := kernLd64y(x, 0), kernLd64y(x, 8)
	mx0, mx1 := mn0, mn1
	n0, n1 := mn0.IsNaN(), mn1.IsNaN()
	mn0, mx0, n0 = kernMinMaxStepY(x, 16, mn0, mx0, n0)
	mn1, mx1, n1 = kernMinMaxStepY(x, 24, mn1, mx1, n1)
	mn0, mx0, n0 = kernMinMaxStepY(x, 32, mn0, mx0, n0)
	mn1, mx1, n1 = kernMinMaxStepY(x, 40, mn1, mx1, n1)
	mn0, mx0, n0 = kernMinMaxStepY(x, 48, mn0, mx0, n0)
	mn1, mx1, n1 = kernMinMaxStepY(x, 56, mn1, mx1, n1)
	mn, mx := mn0.Min(mn1), mx0.Max(mx1)
	mn.GetLo().Min(mn.GetHi()).StoreArray(a)
	mx.GetLo().Max(mx.GetHi()).StoreArray(b)
	nan := n0.Or(n1).ToBits() != 0
	archsimd.ClearAVXUpperBits()
	return nan
}
