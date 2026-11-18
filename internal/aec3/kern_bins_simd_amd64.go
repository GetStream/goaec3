//go:build goexperiment.simd && amd64 && !purego

package aec3

import "simd/archsimd"

// archsimd kernels for the kern_bins.go helpers: 256-bit code with AVX2,
// 128-bit (VEX) code with AVX, the references otherwise.

func kernScale(x []float32, a float32, z []float32) {
	z = z[:len(x)]
	j := 0
	if kernHasAVX2 {
		av := archsimd.BroadcastFloat32x8(a)
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(x[j:]).Mul(av).Store(z[j:])
		}
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
	j := 0
	if kernHasAVX2 {
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(x[j:]).Add(archsimd.LoadFloat32x8(y[j:])).Store(z[j:])
		}
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
	j := 0
	if kernHasAVX2 {
		for ; j+8 <= len(x); j += 8 {
			archsimd.LoadFloat32x8(x[j:]).Sub(archsimd.LoadFloat32x8(y[j:])).Store(z[j:])
		}
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
		kernSumRowsGeneric(dst, rows)
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
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[8]float32)(v[2*k : 2*k+8])
		a := archsimd.LoadFloat32x4Array((*[4]float32)(p[0:4]))
		b := archsimd.LoadFloat32x4Array((*[4]float32)(p[4:8]))
		a.ConcatPermuteScalars(0, 2, 4, 6, b).StoreArray((*[4]float32)(re[k : k+4]))
		a.ConcatPermuteScalars(1, 3, 5, 7, b).StoreArray((*[4]float32)(im[k : k+4]))
	}
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
	for k := 0; k < FftLengthBy2; k += 4 {
		a := archsimd.LoadFloat32x4Array((*[4]float32)(re[k : k+4])).ToBits()
		b := archsimd.LoadFloat32x4Array((*[4]float32)(im[k : k+4])).ToBits()
		p := (*[8]float32)(v[2*k : 2*k+8])
		a.InterleaveLo(b).BitsToFloat32().StoreArray((*[4]float32)(p[0:4]))
		a.InterleaveHi(b).BitsToFloat32().StoreArray((*[4]float32)(p[4:8]))
	}
	v[1] = re[FftLengthBy2]
}

// kernMinMaxFast reports whether kernMinMax has a fast path.
const kernMinMaxFast = false

// kernMinMax defers to the caller's loop (MINPS/MAXPS do not propagate NaN).
func kernMinMax(x []float32) (mn, mx float32, ok bool) { return 0, 0, false }

// kernClamp is kernClampGeneric, std::min(std::max(x, lo), hi), as compares
// and selects: MINPS/MAXPS return their second operand on NaN and equal
// zeros, and archsimd does not fix which operand that is.
func kernClamp(x []float32, lo, hi float32) {
	j := 0
	if kernHasAVX2 {
		lv, hv := archsimd.BroadcastFloat32x8(lo), archsimd.BroadcastFloat32x8(hi)
		for ; j+8 <= len(x); j += 8 {
			v := archsimd.LoadFloat32x8(x[j:])
			v = lv.IfElse(v.Less(lv), v)
			hv.IfElse(hv.Less(v), v).Store(x[j:])
		}
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
