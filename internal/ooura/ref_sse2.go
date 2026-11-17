package ooura

// Pure-Go transcription of ooura_fft_sse2.cc, intrinsic by intrinsic. It is
// the specification the amd64 assembly is tested against. Per lane it
// computes the same operations as the NEON lane (multiplying by the
// {-1,1,-1,1} sign table is exact, so vmlaq(x, y, s) == x + s*y), which
// TestSSE2RefMatchesNeonRef checks.

// shufpsZYXW(a, b) is _mm_shuffle_ps(a, b, _MM_SHUFFLE(Z, Y, X, W)) and
// pshufdZYXW(a) is _mm_shuffle_epi32(a, _MM_SHUFFLE(Z, Y, X, W)), one
// function per immediate the C code uses.

func shufps0123(a, b f32x4) f32x4 { return f32x4{a.e3, a.e2, b.e1, b.e0} }

func shufps0202(a, b f32x4) f32x4 { return f32x4{a.e2, a.e0, b.e2, b.e0} }

func shufps1010(a, b f32x4) f32x4 { return f32x4{a.e0, a.e1, b.e0, b.e1} }

func shufps1032(a, b f32x4) f32x4 { return f32x4{a.e2, a.e3, b.e0, b.e1} }

func shufps1313(a, b f32x4) f32x4 { return f32x4{a.e3, a.e1, b.e3, b.e1} }

func shufps2020(a, b f32x4) f32x4 { return f32x4{a.e0, a.e2, b.e0, b.e2} }

func shufps2222(a, b f32x4) f32x4 { return f32x4{a.e2, a.e2, b.e2, b.e2} }

func shufps2301(a, b f32x4) f32x4 { return f32x4{a.e1, a.e0, b.e3, b.e2} }

func shufps3131(a, b f32x4) f32x4 { return f32x4{a.e1, a.e3, b.e1, b.e3} }

func shufps3232(a, b f32x4) f32x4 { return f32x4{a.e2, a.e3, b.e2, b.e3} }

func shufps3333(a, b f32x4) f32x4 { return f32x4{a.e3, a.e3, b.e3, b.e3} }

func pshufd2301(a f32x4) f32x4 { return f32x4{a.e1, a.e0, a.e3, a.e2} }

func pshufd2323(a f32x4) f32x4 { return f32x4{a.e3, a.e2, a.e3, a.e2} }

func pshufd3232(a f32x4) f32x4 { return f32x4{a.e2, a.e3, a.e2, a.e3} }

func mmUnpacklo(a, b f32x4) f32x4 { return f32x4{a.e0, b.e0, a.e1, b.e1} }

func mmUnpackhi(a, b f32x4) f32x4 { return f32x4{a.e2, b.e2, a.e3, b.e3} }

// mmLoadl is _mm_loadl_epi64: two floats in the low half, zeros above.
func mmLoadl(a []float32) f32x4 { return f32x4{a[0], a[1], 0, 0} }

// mmStorel is _mm_storel_epi64.
func mmStorel(a []float32, v f32x4) { a[0], a[1] = v.e0, v.e1 }

func fftSSE2Ref(a *[128]float32) {
	bitrv2(a)
	cft1stSSE2Ref(a)
	cftmdlSSE2Ref(a)
	cftfsubTail(a)
	rftfsubSSE2Ref(a)
	xi := a[0] - a[1]
	a[0] += a[1]
	a[1] = xi
}

func inverseFftSSE2Ref(a *[128]float32) {
	a[1] = float32(0.5 * (a[0] - a[1]))
	a[0] -= a[1]
	rftbsubSSE2Ref(a)
	bitrv2(a)
	cft1stSSE2Ref(a)
	cftmdlSSE2Ref(a)
	cftbsubTail(a)
}

func cft1stSSE2Ref(a *[128]float32) {
	mmSwapSign := arr4(&swapSign)
	for j, k2 := 0, 0; j < 128; j, k2 = j+16, k2+4 {
		a00v := ld4(a[j+0:])
		a04v := ld4(a[j+4:])
		a08v := ld4(a[j+8:])
		a12v := ld4(a[j+12:])
		a01v := shufps1010(a00v, a08v)
		a23v := shufps3232(a00v, a08v)
		a45v := shufps1010(a04v, a12v)
		a67v := shufps3232(a04v, a12v)
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
		a01v = vadd(x0v, x2v)
		x0v = vsub(x0v, x2v)
		x0w := shufps2301(x0v, x0v)
		a45v = vadd(vmul(wk2rv, x0v), vmul(wk2iv, x0w))
		x3w := shufps2301(x3v, x3v)
		x3s := vmul(mmSwapSign, x3w)
		x0v = vadd(x1v, x3s)
		x0w = shufps2301(x0v, x0v)
		a23v = vadd(vmul(wk1rv, x0v), vmul(wk1iv, x0w))
		x0v = vsub(x1v, x3s)
		x0w = shufps2301(x0v, x0v)
		a67v = vadd(vmul(wk3rv, x0v), vmul(wk3iv, x0w))
		st4(a[j+0:], shufps1010(a01v, a23v))
		st4(a[j+4:], shufps1010(a45v, a67v))
		st4(a[j+8:], shufps3232(a01v, a23v))
		st4(a[j+12:], shufps3232(a45v, a67v))
	}
}

func cftmdlSSE2Ref(a *[128]float32) {
	const l = 8
	mmSwapSign := arr4(&swapSign)
	wk1rv := arr4(&cftmdlWk1r)
	for j0 := 0; j0 < l; j0 += 2 {
		a0032 := shufps1010(mmLoadl(a[j0+0:]), mmLoadl(a[j0+32:]))
		a0840 := shufps1010(mmLoadl(a[j0+8:]), mmLoadl(a[j0+40:]))
		x0 := vadd(a0032, a0840)
		x1 := vsub(a0032, a0840)
		a1648 := shufps1010(mmLoadl(a[j0+16:]), mmLoadl(a[j0+48:]))
		a2456 := shufps1010(mmLoadl(a[j0+24:]), mmLoadl(a[j0+56:]))
		x2 := vadd(a1648, a2456)
		x3 := vsub(a1648, a2456)
		xx0 := vadd(x0, x2)
		xx1 := vsub(x0, x2)
		x3i := pshufd2301(x3)
		x3Swapped := vmul(mmSwapSign, x3i)
		x1x3Add := vadd(x1, x3Swapped)
		x1x3Sub := vsub(x1, x3Swapped)
		yy0 := shufps2222(x1x3Add, x1x3Sub)
		yy1 := shufps3333(x1x3Add, x1x3Sub)
		yy2 := vmul(mmSwapSign, yy1)
		yy3 := vadd(yy0, yy2)
		yy4 := vmul(wk1rv, yy3)
		mmStorel(a[j0+0:], xx0)
		mmStorel(a[j0+32:], pshufd3232(xx0))
		mmStorel(a[j0+16:], xx1)
		mmStorel(a[j0+48:], pshufd2323(xx1))
		a[j0+48] = -a[j0+48]
		mmStorel(a[j0+8:], x1x3Add)
		mmStorel(a[j0+24:], x1x3Sub)
		mmStorel(a[j0+40:], yy4)
		mmStorel(a[j0+56:], pshufd2323(yy4))
	}
	const k, k2 = 64, 4
	wk2rv := ld4(rdftWk2r[k2:])
	wk2iv := ld4(rdftWk2i[k2:])
	wk1iv := ld4(rdftWk1i[k2:])
	wk3rv := ld4(rdftWk3r[k2:])
	wk3iv := ld4(rdftWk3i[k2:])
	wk1rv = ld4(rdftWk1r[k2:])
	for j0 := k; j0 < l+k; j0 += 2 {
		a0032 := shufps1010(mmLoadl(a[j0+0:]), mmLoadl(a[j0+32:]))
		a0840 := shufps1010(mmLoadl(a[j0+8:]), mmLoadl(a[j0+40:]))
		x0 := vadd(a0032, a0840)
		x1 := vsub(a0032, a0840)
		a1648 := shufps1010(mmLoadl(a[j0+16:]), mmLoadl(a[j0+48:]))
		a2456 := shufps1010(mmLoadl(a[j0+24:]), mmLoadl(a[j0+56:]))
		x2 := vadd(a1648, a2456)
		x3 := vsub(a1648, a2456)
		xx := vadd(x0, x2)
		xx1 := vsub(x0, x2)
		xx2 := vmul(xx1, wk2rv)
		xx3 := vmul(wk2iv, pshufd2301(xx1))
		xx4 := vadd(xx2, xx3)
		x3i := pshufd2301(x3)
		x3Swapped := vmul(mmSwapSign, x3i)
		x1x3Add := vadd(x1, x3Swapped)
		x1x3Sub := vsub(x1, x3Swapped)
		xx10 := vmul(x1x3Add, wk1rv)
		xx11 := vmul(wk1iv, pshufd2301(x1x3Add))
		xx12 := vadd(xx10, xx11)
		xx20 := vmul(x1x3Sub, wk3rv)
		xx21 := vmul(wk3iv, pshufd2301(x1x3Sub))
		xx22 := vadd(xx20, xx21)
		mmStorel(a[j0+0:], xx)
		mmStorel(a[j0+32:], pshufd3232(xx))
		mmStorel(a[j0+16:], xx4)
		mmStorel(a[j0+48:], pshufd3232(xx4))
		mmStorel(a[j0+8:], xx12)
		mmStorel(a[j0+40:], pshufd3232(xx12))
		mmStorel(a[j0+24:], xx22)
		mmStorel(a[j0+56:], pshufd3232(xx22))
	}
}

// rftSSE2Ref is the shared body of rftfsub_128_SSE2 (inverse=false) and
// rftbsub_128_SSE2 (inverse=true).
func rftSSE2Ref(a *[128]float32, inverse bool) {
	c := rdftW[32:]
	mmHalf := dup4(0.5)
	j1, j2 := 1, 2
	for ; j2+7 < 64; j1, j2 = j1+4, j2+8 {
		cj1 := ld4(c[j1:])
		ck1 := ld4(c[29-j1:])
		wkrt := vsub(mmHalf, ck1)
		wkr := shufps0123(wkrt, wkrt)
		wki := cj1
		aj20 := ld4(a[0+j2:])
		aj24 := ld4(a[4+j2:])
		ak20 := ld4(a[122-j2:])
		ak24 := ld4(a[126-j2:])
		aj2p0 := shufps2020(aj20, aj24)
		aj2p1 := shufps3131(aj20, aj24)
		ak2p0 := shufps0202(ak24, ak20)
		ak2p1 := shufps1313(ak24, ak20)
		xr := vsub(aj2p0, ak2p0)
		xi := vadd(aj2p1, ak2p1)
		a_ := vmul(wkr, xr)
		b_ := vmul(wki, xi)
		c_ := vmul(wkr, xi)
		d_ := vmul(wki, xr)
		var yr, yi, aj2p0n, aj2p1n, ak2p0n, ak2p1n f32x4
		if !inverse {
			yr = vsub(a_, b_)
			yi = vadd(c_, d_)
			aj2p0n = vsub(aj2p0, yr)
			aj2p1n = vsub(aj2p1, yi)
			ak2p0n = vadd(ak2p0, yr)
			ak2p1n = vsub(ak2p1, yi)
		} else {
			yr = vadd(a_, b_)
			yi = vsub(c_, d_)
			aj2p0n = vsub(aj2p0, yr)
			aj2p1n = vsub(yi, aj2p1)
			ak2p0n = vadd(ak2p0, yr)
			ak2p1n = vsub(yi, ak2p1)
		}
		aj20n := mmUnpacklo(aj2p0n, aj2p1n)
		aj24n := mmUnpackhi(aj2p0n, aj2p1n)
		ak20nt := mmUnpackhi(ak2p0n, ak2p1n)
		ak24nt := mmUnpacklo(ak2p0n, ak2p1n)
		ak20n := shufps1032(ak20nt, ak20nt)
		ak24n := shufps1032(ak24nt, ak24nt)
		st4(a[0+j2:], aj20n)
		st4(a[4+j2:], aj24n)
		st4(a[122-j2:], ak20n)
		st4(a[126-j2:], ak24n)
	}
	for ; j2 < 64; j1, j2 = j1+1, j2+2 {
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2+0] - a[k2+0]
		xi := a[j2+1] + a[k2+1]
		if !inverse {
			yr := float32(wkr*xr) - float32(wki*xi)
			yi := float32(wkr*xi) + float32(wki*xr)
			a[j2+0] -= yr
			a[j2+1] -= yi
			a[k2+0] += yr
			a[k2+1] -= yi
		} else {
			yr := float32(wkr*xr) + float32(wki*xi)
			yi := float32(wkr*xi) - float32(wki*xr)
			a[j2+0] = a[j2+0] - yr
			a[j2+1] = yi - a[j2+1]
			a[k2+0] = yr + a[k2+0]
			a[k2+1] = yi - a[k2+1]
		}
	}
}

func rftfsubSSE2Ref(a *[128]float32) { rftSSE2Ref(a, false) }

func rftbsubSSE2Ref(a *[128]float32) {
	a[1] = -a[1]
	rftSSE2Ref(a, true)
	a[65] = -a[65]
}
