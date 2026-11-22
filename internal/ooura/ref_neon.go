package ooura

// The transcription of ooura_fft_neon.cc, the spec of the NEON lane. Each
// helper is one NEON intrinsic on a vector of four float32. vmlaq and vmlsq
// are unfused: they round the product before the addition, so every product
// is wrapped in float32().

// f32x4 is a struct rather than an array so the compiler keeps the lanes
// in registers.
type f32x4 struct{ e0, e1, e2, e3 float32 }

func arr4(a *[4]float32) f32x4 { return f32x4{a[0], a[1], a[2], a[3]} }

func ld4(a []float32) f32x4 { return f32x4{a[0], a[1], a[2], a[3]} }

func st4(a []float32, v f32x4) { a[0], a[1], a[2], a[3] = v.e0, v.e1, v.e2, v.e3 }

func vadd(a, b f32x4) f32x4 { return f32x4{a.e0 + b.e0, a.e1 + b.e1, a.e2 + b.e2, a.e3 + b.e3} }

func vsub(a, b f32x4) f32x4 { return f32x4{a.e0 - b.e0, a.e1 - b.e1, a.e2 - b.e2, a.e3 - b.e3} }

func vmul(a, b f32x4) f32x4 {
	return f32x4{float32(a.e0 * b.e0), float32(a.e1 * b.e1), float32(a.e2 * b.e2), float32(a.e3 * b.e3)}
}

// vmla is vmlaq_f32(a, b, c) = a + b*c with the product rounded first.
func vmla(a, b, c f32x4) f32x4 { return vadd(a, vmul(b, c)) }

// vmls is vmlsq_f32(a, b, c) = a - b*c with the product rounded first.
func vmls(a, b, c f32x4) f32x4 { return vsub(a, vmul(b, c)) }

// vrev64 swaps the two lanes inside each 64-bit half.
func vrev64(a f32x4) f32x4 { return f32x4{a.e1, a.e0, a.e3, a.e2} }

// vcombineLo is vcombine_f32(vget_low_f32(a), vget_low_f32(b)).
func vcombineLo(a, b f32x4) f32x4 { return f32x4{a.e0, a.e1, b.e0, b.e1} }

// vcombineHi is vcombine_f32(vget_high_f32(a), vget_high_f32(b)).
func vcombineHi(a, b f32x4) f32x4 { return f32x4{a.e2, a.e3, b.e2, b.e3} }

// ld2x2 is vcombine_f32(vld1_f32(p), vld1_f32(q)).
func ld2x2(a []float32, p, q int) f32x4 { return f32x4{a[p], a[p+1], a[q], a[q+1]} }

func stLo(a []float32, p int, v f32x4) { a[p], a[p+1] = v.e0, v.e1 }

func stHi(a []float32, p int, v f32x4) { a[p], a[p+1] = v.e2, v.e3 }

// reverseOrder is reverse_order_f32x4: A B C D -> D C B A.
func reverseOrder(a f32x4) f32x4 { return f32x4{a.e3, a.e2, a.e1, a.e0} }

// vld2 deinterleaves eight floats into even and odd lanes.
func vld2(a []float32) (f32x4, f32x4) {
	return f32x4{a[0], a[2], a[4], a[6]}, f32x4{a[1], a[3], a[5], a[7]}
}

// vst2 interleaves two vectors into eight floats.
func vst2(a []float32, v0, v1 f32x4) {
	a[0], a[1], a[2], a[3] = v0.e0, v1.e0, v0.e1, v1.e1
	a[4], a[5], a[6], a[7] = v0.e2, v1.e2, v0.e3, v1.e3
}

// vzip is vzipq_f32: interleave the low halves, then the high halves.
func vzip(a, b f32x4) (f32x4, f32x4) {
	return f32x4{a.e0, b.e0, a.e1, b.e1}, f32x4{a.e2, b.e2, a.e3, b.e3}
}

func dup4(x float32) f32x4 { return f32x4{x, x, x, x} }

func fftNeonRef(a *[128]float32) {
	bitrv2(a)
	cft1stNeonRef(a)
	cftmdlNeonRef(a)
	cftfsubTail(a)
	rftfsubNeonRef(a)
	xi := a[0] - a[1]
	a[0] += a[1]
	a[1] = xi
}

func inverseFftNeonRef(a *[128]float32) {
	a[1] = float32(0.5 * (a[0] - a[1]))
	a[0] -= a[1]
	rftbsubNeonRef(a)
	bitrv2(a)
	cft1stNeonRef(a)
	cftmdlNeonRef(a)
	cftbsubTail(a)
}

func cft1stNeonRef(a *[128]float32) {
	vecSwapSign := arr4(&swapSign)
	for j, k2 := 0, 0; j < 128; j, k2 = j+16, k2+4 {
		a00v := ld4(a[j+0:])
		a04v := ld4(a[j+4:])
		a08v := ld4(a[j+8:])
		a12v := ld4(a[j+12:])
		a01v := vcombineLo(a00v, a08v)
		a23v := vcombineHi(a00v, a08v)
		a45v := vcombineLo(a04v, a12v)
		a67v := vcombineHi(a04v, a12v)
		wk1rv := ld4(rdftWk1r[k2:])
		wk1iv := ld4(rdftWk1i[k2:])
		wk2rv := ld4(rdftWk2r[k2:])
		wk2iv := ld4(rdftWk2i[k2:])
		wk3rv := ld4(rdftWk3r[k2:])
		wk3iv := ld4(rdftWk3i[k2:])
		x0v := vadd(a01v, a23v)
		x1v := vsub(a01v, a23v)
		x2v := vadd(a45v, a67v)
		x3v := vsub(a45v, a67v)
		x3w := vrev64(x3v)
		a01v = vadd(x0v, x2v)
		x0v = vsub(x0v, x2v)
		x0w := vrev64(x0v)
		a45v = vmul(wk2rv, x0v)
		a45v = vmla(a45v, wk2iv, x0w)
		x0v = vmla(x1v, x3w, vecSwapSign)
		x0w = vrev64(x0v)
		a23v = vmul(wk1rv, x0v)
		a23v = vmla(a23v, wk1iv, x0w)
		x0v = vmls(x1v, x3w, vecSwapSign)
		x0w = vrev64(x0v)
		a67v = vmul(wk3rv, x0v)
		a67v = vmla(a67v, wk3iv, x0w)
		st4(a[j+0:], vcombineLo(a01v, a23v))
		st4(a[j+4:], vcombineLo(a45v, a67v))
		st4(a[j+8:], vcombineHi(a01v, a23v))
		st4(a[j+12:], vcombineHi(a45v, a67v))
	}
}

func cftmdlNeonRef(a *[128]float32) {
	const l = 8
	vecSwapSign := arr4(&swapSign)
	wk1rv := arr4(&cftmdlWk1r)
	for j := 0; j < l; j += 2 {
		a0032 := ld2x2(a[:], j+0, j+32)
		a0840 := ld2x2(a[:], j+8, j+40)
		x0 := vadd(a0032, a0840)
		x1 := vsub(a0032, a0840)
		a1648 := ld2x2(a[:], j+16, j+48)
		a2456 := ld2x2(a[:], j+24, j+56)
		x2 := vadd(a1648, a2456)
		x3 := vsub(a1648, a2456)
		xx0 := vadd(x0, x2)
		xx1 := vsub(x0, x2)
		x3rev := vrev64(x3)
		x1x3Add := vmla(x1, vecSwapSign, x3rev)
		x1x3Sub := vmls(x1, vecSwapSign, x3rev)
		yy0as := f32x4{x1x3Add.e2, x1x3Add.e2, x1x3Sub.e2, x1x3Sub.e2}
		yy1as := f32x4{x1x3Add.e3, x1x3Add.e3, x1x3Sub.e3, x1x3Sub.e3}
		yy0 := vmla(yy0as, vecSwapSign, yy1as)
		yy4 := vmul(wk1rv, yy0)
		xx1Rev := vrev64(xx1)
		yy4Rev := vrev64(yy4)
		stLo(a[:], j+0, xx0)
		stHi(a[:], j+32, xx0)
		stLo(a[:], j+16, xx1)
		stHi(a[:], j+48, xx1Rev)
		a[j+48] = -a[j+48]
		stLo(a[:], j+8, x1x3Add)
		stLo(a[:], j+24, x1x3Sub)
		stLo(a[:], j+40, yy4)
		stHi(a[:], j+56, yy4Rev)
	}
	const k, k2 = 64, 4
	wk2rv := ld4(rdftWk2r[k2:])
	wk2iv := ld4(rdftWk2i[k2:])
	wk1iv := ld4(rdftWk1i[k2:])
	wk3rv := ld4(rdftWk3r[k2:])
	wk3iv := ld4(rdftWk3i[k2:])
	wk1rv = ld4(rdftWk1r[k2:])
	for j := k; j < l+k; j += 2 {
		a0032 := ld2x2(a[:], j+0, j+32)
		a0840 := ld2x2(a[:], j+8, j+40)
		x0 := vadd(a0032, a0840)
		x1 := vsub(a0032, a0840)
		a1648 := ld2x2(a[:], j+16, j+48)
		a2456 := ld2x2(a[:], j+24, j+56)
		x2 := vadd(a1648, a2456)
		x3 := vsub(a1648, a2456)
		xx := vadd(x0, x2)
		xx1 := vsub(x0, x2)
		x3rev := vrev64(x3)
		x1x3Add := vmla(x1, vecSwapSign, x3rev)
		x1x3Sub := vmls(x1, vecSwapSign, x3rev)
		xx4 := vmul(wk2rv, xx1)
		xx12 := vmul(wk1rv, x1x3Add)
		xx22 := vmul(wk3rv, x1x3Sub)
		xx4 = vmla(xx4, wk2iv, vrev64(xx1))
		xx12 = vmla(xx12, wk1iv, vrev64(x1x3Add))
		xx22 = vmla(xx22, wk3iv, vrev64(x1x3Sub))
		stLo(a[:], j+0, xx)
		stHi(a[:], j+32, xx)
		stLo(a[:], j+16, xx4)
		stHi(a[:], j+48, xx4)
		stLo(a[:], j+8, xx12)
		stHi(a[:], j+40, xx12)
		stLo(a[:], j+24, xx22)
		stHi(a[:], j+56, xx22)
	}
}

func rftfsubNeonRef(a *[128]float32) {
	c := rdftW[32:]
	mmHalf := dup4(0.5)
	j1, j2 := 1, 2
	for ; j2+7 < 64; j1, j2 = j1+4, j2+8 {
		cj1 := ld4(c[j1:])
		ck1 := ld4(c[29-j1:])
		wkrt := vsub(mmHalf, ck1)
		wkr := reverseOrder(wkrt)
		wki := cj1
		aj2p0, aj2p1 := vld2(a[j2:])
		k20, k21 := vld2(a[122-j2:])
		ak2p0 := reverseOrder(k20)
		ak2p1 := reverseOrder(k21)
		xr := vsub(aj2p0, ak2p0)
		xi := vadd(aj2p1, ak2p1)
		a_ := vmul(wkr, xr)
		b_ := vmul(wki, xi)
		c_ := vmul(wkr, xi)
		d_ := vmul(wki, xr)
		yr := vsub(a_, b_)
		yi := vadd(c_, d_)
		ak2p0n := vadd(ak2p0, yr)
		ak2p1n := vsub(ak2p1, yi)
		ak2p0nr := vrev64(ak2p0n)
		ak2p1nr := vrev64(ak2p1n)
		ak2n0, ak2n1 := vzip(ak2p0nr, ak2p1nr)
		aj2p0 = vsub(aj2p0, yr)
		aj2p1 = vsub(aj2p1, yi)
		vst2(a[j2:], aj2p0, aj2p1)
		st4(a[122-j2:], ak2n1)
		st4(a[126-j2:], ak2n0)
	}
	for ; j2 < 64; j1, j2 = j1+1, j2+2 {
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2+0] - a[k2+0]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) - float32(wki*xi)
		yi := float32(wkr*xi) + float32(wki*xr)
		a[j2+0] -= yr
		a[j2+1] -= yi
		a[k2+0] += yr
		a[k2+1] -= yi
	}
}

func rftbsubNeonRef(a *[128]float32) {
	c := rdftW[32:]
	mmHalf := dup4(0.5)
	a[1] = -a[1]
	j1, j2 := 1, 2
	for ; j2+7 < 64; j1, j2 = j1+4, j2+8 {
		cj1 := ld4(c[j1:])
		ck1 := ld4(c[29-j1:])
		wkrt := vsub(mmHalf, ck1)
		wkr := reverseOrder(wkrt)
		wki := cj1
		aj2p0, aj2p1 := vld2(a[j2:])
		k20, k21 := vld2(a[122-j2:])
		ak2p0 := reverseOrder(k20)
		ak2p1 := reverseOrder(k21)
		xr := vsub(aj2p0, ak2p0)
		xi := vadd(aj2p1, ak2p1)
		a_ := vmul(wkr, xr)
		b_ := vmul(wki, xi)
		c_ := vmul(wkr, xi)
		d_ := vmul(wki, xr)
		yr := vadd(a_, b_)
		yi := vsub(c_, d_)
		ak2p0n := vadd(ak2p0, yr)
		ak2p1n := vsub(yi, ak2p1)
		ak2p0nr := vrev64(ak2p0n)
		ak2p1nr := vrev64(ak2p1n)
		ak2n0, ak2n1 := vzip(ak2p0nr, ak2p1nr)
		aj2p0 = vsub(aj2p0, yr)
		aj2p1 = vsub(yi, aj2p1)
		vst2(a[j2:], aj2p0, aj2p1)
		st4(a[122-j2:], ak2n1)
		st4(a[126-j2:], ak2n0)
	}
	for ; j2 < 64; j1, j2 = j1+1, j2+2 {
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2+0] - a[k2+0]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) + float32(wki*xi)
		yi := float32(wkr*xi) - float32(wki*xr)
		a[j2+0] = a[j2+0] - yr
		a[j2+1] = yi - a[j2+1]
		a[k2+0] = yr + a[k2+0]
		a[k2+1] = yi - a[k2+1]
	}
	a[65] = -a[65]
}
