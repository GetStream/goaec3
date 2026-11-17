// Package ooura implements the fixed-size 128-point real FFT that WebRTC AEC3
// uses (common_audio/third_party/ooura/fft_size_128/ooura_fft.cc).
//
// Every product is wrapped in an explicit float32 conversion so the Go
// compiler never contracts a multiply-add into an FMA: the reference C and
// NEON paths round each product separately (vmlaq_f32 is unfused).
package ooura

// Fft computes the forward real FFT of a in place. The output packs
// a[0]=Re(X0), a[1]=Re(X64), then interleaved Re/Im for bins 1..63.
func Fft(a *[128]float32) {
	bitrv2(a)
	cftfsub(a)
	rftfsub(a)
	xi := a[0] - a[1]
	a[0] += a[1]
	a[1] = xi
}

// InverseFft computes the unscaled inverse real FFT of a in place.
func InverseFft(a *[128]float32) {
	a[1] = float32(0.5 * (a[0] - a[1]))
	a[0] -= a[1]
	rftbsub(a)
	bitrv2(a)
	cftbsub(a)
}

func cftfsub(a *[128]float32) {
	cft1st(a)
	cftmdl(a)
	const l = 32
	for j := 0; j < l; j += 2 {
		j1 := j + l
		j2 := j1 + l
		j3 := j2 + l
		x0r := a[j] + a[j1]
		x0i := a[j+1] + a[j1+1]
		x1r := a[j] - a[j1]
		x1i := a[j+1] - a[j1+1]
		x2r := a[j2] + a[j3]
		x2i := a[j2+1] + a[j3+1]
		x3r := a[j2] - a[j3]
		x3i := a[j2+1] - a[j3+1]
		a[j] = x0r + x2r
		a[j+1] = x0i + x2i
		a[j2] = x0r - x2r
		a[j2+1] = x0i - x2i
		a[j1] = x1r - x3i
		a[j1+1] = x1i + x3r
		a[j3] = x1r + x3i
		a[j3+1] = x1i - x3r
	}
}

func cftbsub(a *[128]float32) {
	cft1st(a)
	cftmdl(a)
	const l = 32
	for j := 0; j < l; j += 2 {
		j1 := j + l
		j2 := j1 + l
		j3 := j2 + l
		x0r := a[j] + a[j1]
		x0i := -a[j+1] - a[j1+1]
		x1r := a[j] - a[j1]
		x1i := -a[j+1] + a[j1+1]
		x2r := a[j2] + a[j3]
		x2i := a[j2+1] + a[j3+1]
		x3r := a[j2] - a[j3]
		x3i := a[j2+1] - a[j3+1]
		a[j] = x0r + x2r
		a[j+1] = x0i - x2i
		a[j2] = x0r - x2r
		a[j2+1] = x0i + x2i
		a[j1] = x1r - x3i
		a[j1+1] = x1i - x3r
		a[j3] = x1r + x3i
		a[j3+1] = x1i + x3r
	}
}

// cmul writes (wr*xr - wi*xi, wr*xi + wi*xr) without FMA contraction.
func cmul(wr, wi, xr, xi float32) (float32, float32) {
	return float32(wr*xr) - float32(wi*xi), float32(wr*xi) + float32(wi*xr)
}

func cft1st(a *[128]float32) {
	x0r := a[0] + a[2]
	x0i := a[1] + a[3]
	x1r := a[0] - a[2]
	x1i := a[1] - a[3]
	x2r := a[4] + a[6]
	x2i := a[5] + a[7]
	x3r := a[4] - a[6]
	x3i := a[5] - a[7]
	a[0] = x0r + x2r
	a[1] = x0i + x2i
	a[4] = x0r - x2r
	a[5] = x0i - x2i
	a[2] = x1r - x3i
	a[3] = x1i + x3r
	a[6] = x1r + x3i
	a[7] = x1i - x3r
	wk1r := rdftW[2]
	x0r = a[8] + a[10]
	x0i = a[9] + a[11]
	x1r = a[8] - a[10]
	x1i = a[9] - a[11]
	x2r = a[12] + a[14]
	x2i = a[13] + a[15]
	x3r = a[12] - a[14]
	x3i = a[13] - a[15]
	a[8] = x0r + x2r
	a[9] = x0i + x2i
	a[12] = x2i - x0i
	a[13] = x0r - x2r
	x0r = x1r - x3i
	x0i = x1i + x3r
	a[10] = wk1r * (x0r - x0i)
	a[11] = wk1r * (x0r + x0i)
	x0r = x3i + x1r
	x0i = x3r - x1i
	a[14] = wk1r * (x0i - x0r)
	a[15] = wk1r * (x0i + x0r)
	k1 := 0
	for j := 16; j < 128; j += 16 {
		k1 += 2
		k2 := 2 * k1
		wk2r, wk2i := rdftW[k1], rdftW[k1+1]
		wk1r, wk1i := rdftW[k2], rdftW[k2+1]
		wk3r, wk3i := rdftWk3riFirst[k1], rdftWk3riFirst[k1+1]
		x0r := a[j+0] + a[j+2]
		x0i := a[j+1] + a[j+3]
		x1r := a[j+0] - a[j+2]
		x1i := a[j+1] - a[j+3]
		x2r := a[j+4] + a[j+6]
		x2i := a[j+5] + a[j+7]
		x3r := a[j+4] - a[j+6]
		x3i := a[j+5] - a[j+7]
		a[j+0] = x0r + x2r
		a[j+1] = x0i + x2i
		x0r -= x2r
		x0i -= x2i
		a[j+4], a[j+5] = cmul(wk2r, wk2i, x0r, x0i)
		a[j+2], a[j+3] = cmul(wk1r, wk1i, x1r-x3i, x1i+x3r)
		a[j+6], a[j+7] = cmul(wk3r, wk3i, x1r+x3i, x1i-x3r)
		wk1r, wk1i = rdftW[k2+2], rdftW[k2+3]
		wk3r, wk3i = rdftWk3riSecond[k1], rdftWk3riSecond[k1+1]
		x0r = a[j+8] + a[j+10]
		x0i = a[j+9] + a[j+11]
		x1r = a[j+8] - a[j+10]
		x1i = a[j+9] - a[j+11]
		x2r = a[j+12] + a[j+14]
		x2i = a[j+13] + a[j+15]
		x3r = a[j+12] - a[j+14]
		x3i = a[j+13] - a[j+15]
		a[j+8] = x0r + x2r
		a[j+9] = x0i + x2i
		x0r -= x2r
		x0i -= x2i
		a[j+12] = float32(-wk2i*x0r) - float32(wk2r*x0i)
		a[j+13] = float32(-wk2i*x0i) + float32(wk2r*x0r)
		a[j+10], a[j+11] = cmul(wk1r, wk1i, x1r-x3i, x1i+x3r)
		a[j+14], a[j+15] = cmul(wk3r, wk3i, x1r+x3i, x1i-x3r)
	}
}

func cftmdl(a *[128]float32) {
	const l, n, m = 8, 128, 32
	for j0 := 0; j0 < l; j0 += 2 {
		j1, j2, j3 := j0+8, j0+16, j0+24
		x0r := a[j0] + a[j1]
		x0i := a[j0+1] + a[j1+1]
		x1r := a[j0] - a[j1]
		x1i := a[j0+1] - a[j1+1]
		x2r := a[j2] + a[j3]
		x2i := a[j2+1] + a[j3+1]
		x3r := a[j2] - a[j3]
		x3i := a[j2+1] - a[j3+1]
		a[j0] = x0r + x2r
		a[j0+1] = x0i + x2i
		a[j2] = x0r - x2r
		a[j2+1] = x0i - x2i
		a[j1] = x1r - x3i
		a[j1+1] = x1i + x3r
		a[j3] = x1r + x3i
		a[j3+1] = x1i - x3r
	}
	wk1r := rdftW[2]
	for j0 := m; j0 < l+m; j0 += 2 {
		j1, j2, j3 := j0+8, j0+16, j0+24
		x0r := a[j0] + a[j1]
		x0i := a[j0+1] + a[j1+1]
		x1r := a[j0] - a[j1]
		x1i := a[j0+1] - a[j1+1]
		x2r := a[j2] + a[j3]
		x2i := a[j2+1] + a[j3+1]
		x3r := a[j2] - a[j3]
		x3i := a[j2+1] - a[j3+1]
		a[j0] = x0r + x2r
		a[j0+1] = x0i + x2i
		a[j2] = x2i - x0i
		a[j2+1] = x0r - x2r
		x0r = x1r - x3i
		x0i = x1i + x3r
		a[j1] = wk1r * (x0r - x0i)
		a[j1+1] = wk1r * (x0r + x0i)
		x0r = x3i + x1r
		x0i = x3r - x1i
		a[j3] = wk1r * (x0i - x0r)
		a[j3+1] = wk1r * (x0i + x0r)
	}
	k1 := 0
	const m2 = 2 * m
	for k := m2; k < n; k += m2 {
		k1 += 2
		k2 := 2 * k1
		wk2r, wk2i := rdftW[k1], rdftW[k1+1]
		wk1r, wk1i := rdftW[k2], rdftW[k2+1]
		wk3r, wk3i := rdftWk3riFirst[k1], rdftWk3riFirst[k1+1]
		for j0 := k; j0 < l+k; j0 += 2 {
			j1, j2, j3 := j0+8, j0+16, j0+24
			x0r := a[j0] + a[j1]
			x0i := a[j0+1] + a[j1+1]
			x1r := a[j0] - a[j1]
			x1i := a[j0+1] - a[j1+1]
			x2r := a[j2] + a[j3]
			x2i := a[j2+1] + a[j3+1]
			x3r := a[j2] - a[j3]
			x3i := a[j2+1] - a[j3+1]
			a[j0] = x0r + x2r
			a[j0+1] = x0i + x2i
			x0r -= x2r
			x0i -= x2i
			a[j2], a[j2+1] = cmul(wk2r, wk2i, x0r, x0i)
			a[j1], a[j1+1] = cmul(wk1r, wk1i, x1r-x3i, x1i+x3r)
			a[j3], a[j3+1] = cmul(wk3r, wk3i, x1r+x3i, x1i-x3r)
		}
		wk1r, wk1i = rdftW[k2+2], rdftW[k2+3]
		wk3r, wk3i = rdftWk3riSecond[k1], rdftWk3riSecond[k1+1]
		for j0 := k + m; j0 < l+(k+m); j0 += 2 {
			j1, j2, j3 := j0+8, j0+16, j0+24
			x0r := a[j0] + a[j1]
			x0i := a[j0+1] + a[j1+1]
			x1r := a[j0] - a[j1]
			x1i := a[j0+1] - a[j1+1]
			x2r := a[j2] + a[j3]
			x2i := a[j2+1] + a[j3+1]
			x3r := a[j2] - a[j3]
			x3i := a[j2+1] - a[j3+1]
			a[j0] = x0r + x2r
			a[j0+1] = x0i + x2i
			x0r -= x2r
			x0i -= x2i
			a[j2] = float32(-wk2i*x0r) - float32(wk2r*x0i)
			a[j2+1] = float32(-wk2i*x0i) + float32(wk2r*x0r)
			a[j1], a[j1+1] = cmul(wk1r, wk1i, x1r-x3i, x1i+x3r)
			a[j3], a[j3+1] = cmul(wk3r, wk3i, x1r+x3i, x1i-x3r)
		}
	}
}

func rftfsub(a *[128]float32) {
	c := rdftW[32:]
	for j1, j2 := 1, 2; j2 < 64; j1, j2 = j1+1, j2+2 {
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2] - a[k2]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) - float32(wki*xi)
		yi := float32(wkr*xi) + float32(wki*xr)
		a[j2] -= yr
		a[j2+1] -= yi
		a[k2] += yr
		a[k2+1] -= yi
	}
}

func rftbsub(a *[128]float32) {
	c := rdftW[32:]
	a[1] = -a[1]
	for j1, j2 := 1, 2; j2 < 64; j1, j2 = j1+1, j2+2 {
		k2 := 128 - j2
		k1 := 32 - j1
		wkr := 0.5 - c[k1]
		wki := c[j1]
		xr := a[j2] - a[k2]
		xi := a[j2+1] + a[k2+1]
		yr := float32(wkr*xr) + float32(wki*xi)
		yi := float32(wkr*xi) - float32(wki*xr)
		a[j2] = a[j2] - yr
		a[j2+1] = yi - a[j2+1]
		a[k2] = yr + a[k2]
		a[k2+1] = yi - a[k2+1]
	}
	a[65] = -a[65]
}

var bitrvIP = [4]int{0, 64, 32, 96}

func swap2(a *[128]float32, j1, k1 int) {
	a[j1], a[k1] = a[k1], a[j1]
	a[j1+1], a[k1+1] = a[k1+1], a[j1+1]
}

func bitrv2(a *[128]float32) {
	for k := 0; k < 4; k++ {
		for j := 0; j < k; j++ {
			j1 := 2*j + bitrvIP[k]
			k1 := 2*k + bitrvIP[j]
			swap2(a, j1, k1)
			j1 += 8
			k1 += 16
			swap2(a, j1, k1)
			j1 += 8
			k1 -= 8
			swap2(a, j1, k1)
			j1 += 8
			k1 += 16
			swap2(a, j1, k1)
		}
		j1 := 2*k + 8 + bitrvIP[k]
		swap2(a, j1, j1+8)
	}
}
