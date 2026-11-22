package ooura

// Pure-Go FFT of the builds without archsimd kernels.
//
// fftFastLane computes the NEON/SSE2-order transform (the element-wise
// arithmetic of fftNeonRef and fftSSE2Ref, which are identical), and
// fftFastScalar the scalar C order of fftScalar. Both are written for the Go
// compiler rather than as transcriptions:
//
//   - Every float operation is the one of the reference, in the same order
//     per output element; only loads, stores and loop structure differ.
//   - bitrv2 is not a pass. cft1st reads its inputs from their bit-reversed
//     positions and writes a scratch frame; cftmdl reads the scratch frame
//     and writes the caller's frame.
//   - Values live in float32 locals, and every index is a constant offset
//     into a window whose bounds the compiler proves, so the loops carry no
//     bounds checks.
//
// TestFastMatchesRef checks them bit-exactly against the references.

// rev3 reverses three bits.
var rev3 = [8]uint8{0, 4, 2, 6, 1, 5, 3, 7}

func inverseFftFastScalar(a *[128]float32) {
	a[1] = float32(0.5 * (a[0] - a[1]))
	a[0] -= a[1]
	rftbsubFast(a)
	var t [128]float32
	cft1stFastScalar(a, &t)
	cftmdlFastScalar(&t, a)
	cftbsubTailFast(a)
}

// cft1st after bitrv2, both lanes.
//
// bitrv2_128 moves the complex value (float pair) at index rev6(u) to index
// u, where rev6 reverses the six index bits. cft1st_128 works on groups of
// four consecutive pairs: group g = 2q+h (q = block, h = half of the 16-float
// block) holds pairs 4g..4g+3, which bitrv2 took from pairs
// s, s+32, s+16, s+48 with s = 8h + rev3(q). So for s = 0..15 the group reads
// floats 2s + {0,1}, {64,65}, {32,33}, {96,97} of the unpermuted frame and
// writes floats 16q+8h .. 16q+8h+7.

// cft1stFastLane is bitrv2_128 followed by the NEON/SSE2 cft1st_128, from a
// into t. Per group and lane the SIMD code computes, with the twiddle
// vectors wk?[4q+2h], wk?[4q+2h+1]:
//
//	x0 = c0 + c1, x1 = c0 - c1, x2 = c2 + c3, x3 = c2 - c3
//	out0 = x0 + x2
//	y = x0 - x2:                   out2 = wk2r*y + wk2i*swap(y)
//	y = x1 + {-1,1}*swap(x3):      out1 = wk1r*y + wk1i*swap(y)
//	y = x1 - {-1,1}*swap(x3):      out3 = wk3r*y + wk3i*swap(y)
//
// Multiplying by -1 or 1 is exact, so x1 ± {-1,1}*swap(x3) are the plain
// differences and sums below.
func cft1stFastLane(a, t *[128]float32) {
	for s := 0; s < 16; s++ {
		src := (*[98]float32)(a[2*s : 2*s+98])
		q := int(rev3[s&7]) & 7
		h := s >> 3
		d := (*[8]float32)(t[16*q+8*h : 16*q+8*h+8])
		k := 4*q + 2*h
		w1r, w1i := (*[2]float32)(rdftWk1r[k:k+2]), (*[2]float32)(rdftWk1i[k:k+2])
		w2r, w2i := (*[2]float32)(rdftWk2r[k:k+2]), (*[2]float32)(rdftWk2i[k:k+2])
		w3r, w3i := (*[2]float32)(rdftWk3r[k:k+2]), (*[2]float32)(rdftWk3i[k:k+2])
		x0r := src[0] + src[64]
		x0i := src[1] + src[65]
		x1r := src[0] - src[64]
		x1i := src[1] - src[65]
		x2r := src[32] + src[96]
		x2i := src[33] + src[97]
		x3r := src[32] - src[96]
		x3i := src[33] - src[97]
		d[0] = x0r + x2r
		d[1] = x0i + x2i
		yr := x0r - x2r
		yi := x0i - x2i
		d[4] = float32(w2r[0]*yr) + float32(w2i[0]*yi)
		d[5] = float32(w2r[1]*yi) + float32(w2i[1]*yr)
		yr = x1r - x3i
		yi = x1i + x3r
		d[2] = float32(w1r[0]*yr) + float32(w1i[0]*yi)
		d[3] = float32(w1r[1]*yi) + float32(w1i[1]*yr)
		yr = x1r + x3i
		yi = x1i - x3r
		d[6] = float32(w3r[0]*yr) + float32(w3i[0]*yi)
		d[7] = float32(w3r[1]*yi) + float32(w3i[1]*yr)
	}
}

// cftmdlFastLane is the NEON/SSE2 cftmdl_128 from s into a. Lanes 0 and 1 of
// the SIMD vectors are the group at j, lanes 2 and 3 the group at j+32.
func cftmdlFastLane(s, a *[128]float32) {
	w := cftmdlWk1r[0]
	nw := cftmdlWk1r[3]
	for j := 0; j < 8; j += 2 {
		// Group at j: a plain radix-4 butterfly.
		p0, p1, p2, p3 := (*[8]float32)(s[0:8]), (*[8]float32)(s[8:16]), (*[8]float32)(s[16:24]), (*[8]float32)(s[24:32])
		x0r := p0[j] + p1[j]
		x0i := p0[j+1] + p1[j+1]
		x1r := p0[j] - p1[j]
		x1i := p0[j+1] - p1[j+1]
		x2r := p2[j] + p3[j]
		x2i := p2[j+1] + p3[j+1]
		x3r := p2[j] - p3[j]
		x3i := p2[j+1] - p3[j+1]
		d0, d1, d2, d3 := (*[8]float32)(a[0:8]), (*[8]float32)(a[8:16]), (*[8]float32)(a[16:24]), (*[8]float32)(a[24:32])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		d2[j] = x0r - x2r
		d2[j+1] = x0i - x2i
		d1[j] = x1r - x3i
		d1[j+1] = x1i + x3r
		d3[j] = x1r + x3i
		d3[j+1] = x1i - x3r
		// Group at j+32.
		p0, p1, p2, p3 = (*[8]float32)(s[32:40]), (*[8]float32)(s[40:48]), (*[8]float32)(s[48:56]), (*[8]float32)(s[56:64])
		x0r = p0[j] + p1[j]
		x0i = p0[j+1] + p1[j+1]
		x1r = p0[j] - p1[j]
		x1i = p0[j+1] - p1[j+1]
		x2r = p2[j] + p3[j]
		x2i = p2[j+1] + p3[j+1]
		x3r = p2[j] - p3[j]
		x3i = p2[j+1] - p3[j+1]
		d0, d1, d2, d3 = (*[8]float32)(a[32:40]), (*[8]float32)(a[40:48]), (*[8]float32)(a[48:56]), (*[8]float32)(a[56:64])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		d2[j] = -(x0i - x2i)
		d2[j+1] = x0r - x2r
		addr := x1r - x3i
		addi := x1i + x3r
		subr := x1r + x3i
		subi := x1i - x3r
		d1[j] = w * (addr - addi)
		d1[j+1] = w * (addr + addi)
		d3[j] = nw * (subr + subi)
		d3[j+1] = w * (subr - subi)
	}
	const k = 4
	w1r0, w1r1, w1r2, w1r3 := rdftWk1r[k], rdftWk1r[k+1], rdftWk1r[k+2], rdftWk1r[k+3]
	w1i0, w1i1, w1i2, w1i3 := rdftWk1i[k], rdftWk1i[k+1], rdftWk1i[k+2], rdftWk1i[k+3]
	w2r0, w2r1, w2r2, w2r3 := rdftWk2r[k], rdftWk2r[k+1], rdftWk2r[k+2], rdftWk2r[k+3]
	w2i0, w2i1, w2i2, w2i3 := rdftWk2i[k], rdftWk2i[k+1], rdftWk2i[k+2], rdftWk2i[k+3]
	w3r0, w3r1, w3r2, w3r3 := rdftWk3r[k], rdftWk3r[k+1], rdftWk3r[k+2], rdftWk3r[k+3]
	w3i0, w3i1, w3i2, w3i3 := rdftWk3i[k], rdftWk3i[k+1], rdftWk3i[k+2], rdftWk3i[k+3]
	for j := 0; j < 8; j += 2 {
		p0, p1, p2, p3 := (*[8]float32)(s[64:72]), (*[8]float32)(s[72:80]), (*[8]float32)(s[80:88]), (*[8]float32)(s[88:96])
		x0r := p0[j] + p1[j]
		x0i := p0[j+1] + p1[j+1]
		x1r := p0[j] - p1[j]
		x1i := p0[j+1] - p1[j+1]
		x2r := p2[j] + p3[j]
		x2i := p2[j+1] + p3[j+1]
		x3r := p2[j] - p3[j]
		x3i := p2[j+1] - p3[j+1]
		d0, d1, d2, d3 := (*[8]float32)(a[64:72]), (*[8]float32)(a[72:80]), (*[8]float32)(a[80:88]), (*[8]float32)(a[88:96])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		yr := x0r - x2r
		yi := x0i - x2i
		d2[j] = float32(w2r0*yr) + float32(w2i0*yi)
		d2[j+1] = float32(w2r1*yi) + float32(w2i1*yr)
		yr = x1r - x3i
		yi = x1i + x3r
		d1[j] = float32(w1r0*yr) + float32(w1i0*yi)
		d1[j+1] = float32(w1r1*yi) + float32(w1i1*yr)
		yr = x1r + x3i
		yi = x1i - x3r
		d3[j] = float32(w3r0*yr) + float32(w3i0*yi)
		d3[j+1] = float32(w3r1*yi) + float32(w3i1*yr)

		p0, p1, p2, p3 = (*[8]float32)(s[96:104]), (*[8]float32)(s[104:112]), (*[8]float32)(s[112:120]), (*[8]float32)(s[120:128])
		x0r = p0[j] + p1[j]
		x0i = p0[j+1] + p1[j+1]
		x1r = p0[j] - p1[j]
		x1i = p0[j+1] - p1[j+1]
		x2r = p2[j] + p3[j]
		x2i = p2[j+1] + p3[j+1]
		x3r = p2[j] - p3[j]
		x3i = p2[j+1] - p3[j+1]
		d0, d1, d2, d3 = (*[8]float32)(a[96:104]), (*[8]float32)(a[104:112]), (*[8]float32)(a[112:120]), (*[8]float32)(a[120:128])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		yr = x0r - x2r
		yi = x0i - x2i
		d2[j] = float32(w2r2*yr) + float32(w2i2*yi)
		d2[j+1] = float32(w2r3*yi) + float32(w2i3*yr)
		yr = x1r - x3i
		yi = x1i + x3r
		d1[j] = float32(w1r2*yr) + float32(w1i2*yi)
		d1[j+1] = float32(w1r3*yi) + float32(w1i3*yr)
		yr = x1r + x3i
		yi = x1i - x3r
		d3[j] = float32(w3r2*yr) + float32(w3i2*yi)
		d3[j+1] = float32(w3r3*yi) + float32(w3i3*yr)
	}
}

// cft1stFastScalar is bitrv2_128 followed by the scalar cft1st_128, from a
// into t.
func cft1stFastScalar(a, t *[128]float32) {
	// Block 0, first half (s = 0).
	x0r := a[0] + a[64]
	x0i := a[1] + a[65]
	x1r := a[0] - a[64]
	x1i := a[1] - a[65]
	x2r := a[32] + a[96]
	x2i := a[33] + a[97]
	x3r := a[32] - a[96]
	x3i := a[33] - a[97]
	t[0] = x0r + x2r
	t[1] = x0i + x2i
	t[4] = x0r - x2r
	t[5] = x0i - x2i
	t[2] = x1r - x3i
	t[3] = x1i + x3r
	t[6] = x1r + x3i
	t[7] = x1i - x3r
	// Block 0, second half (s = 8).
	wk1r := rdftW[2]
	x0r = a[16] + a[80]
	x0i = a[17] + a[81]
	x1r = a[16] - a[80]
	x1i = a[17] - a[81]
	x2r = a[48] + a[112]
	x2i = a[49] + a[113]
	x3r = a[48] - a[112]
	x3i = a[49] - a[113]
	t[8] = x0r + x2r
	t[9] = x0i + x2i
	t[12] = x2i - x0i
	t[13] = x0r - x2r
	x0r = x1r - x3i
	x0i = x1i + x3r
	t[10] = wk1r * (x0r - x0i)
	t[11] = wk1r * (x0r + x0i)
	x0r = x3i + x1r
	x0i = x3r - x1i
	t[14] = wk1r * (x0i - x0r)
	t[15] = wk1r * (x0i + x0r)
	for s := 1; s < 8; s++ {
		q := int(rev3[s]) & 7 // block j = 16q, k1 = 2q, k2 = 4q
		wk2 := (*[2]float32)(rdftW[2*q : 2*q+2])
		wk1 := (*[4]float32)(rdftW[4*q : 4*q+4])
		wk3f := (*[2]float32)(rdftWk3riFirst[2*q : 2*q+2])
		wk3s := (*[2]float32)(rdftWk3riSecond[2*q : 2*q+2])
		wk2r, wk2i := wk2[0], wk2[1]
		d := (*[16]float32)(t[16*q : 16*q+16])

		src := (*[98]float32)(a[2*s : 2*s+98])
		x0r := src[0] + src[64]
		x0i := src[1] + src[65]
		x1r := src[0] - src[64]
		x1i := src[1] - src[65]
		x2r := src[32] + src[96]
		x2i := src[33] + src[97]
		x3r := src[32] - src[96]
		x3i := src[33] - src[97]
		d[0] = x0r + x2r
		d[1] = x0i + x2i
		x0r -= x2r
		x0i -= x2i
		d[4] = float32(wk2r*x0r) - float32(wk2i*x0i)
		d[5] = float32(wk2r*x0i) + float32(wk2i*x0r)
		x0r = x1r - x3i
		x0i = x1i + x3r
		d[2] = float32(wk1[0]*x0r) - float32(wk1[1]*x0i)
		d[3] = float32(wk1[0]*x0i) + float32(wk1[1]*x0r)
		x0r = x1r + x3i
		x0i = x1i - x3r
		d[6] = float32(wk3f[0]*x0r) - float32(wk3f[1]*x0i)
		d[7] = float32(wk3f[0]*x0i) + float32(wk3f[1]*x0r)

		src = (*[98]float32)((*[112]float32)(a[16:])[2*s : 2*s+98])
		x0r = src[0] + src[64]
		x0i = src[1] + src[65]
		x1r = src[0] - src[64]
		x1i = src[1] - src[65]
		x2r = src[32] + src[96]
		x2i = src[33] + src[97]
		x3r = src[32] - src[96]
		x3i = src[33] - src[97]
		d[8] = x0r + x2r
		d[9] = x0i + x2i
		x0r -= x2r
		x0i -= x2i
		d[12] = float32(-wk2i*x0r) - float32(wk2r*x0i)
		d[13] = float32(-wk2i*x0i) + float32(wk2r*x0r)
		x0r = x1r - x3i
		x0i = x1i + x3r
		d[10] = float32(wk1[2]*x0r) - float32(wk1[3]*x0i)
		d[11] = float32(wk1[2]*x0i) + float32(wk1[3]*x0r)
		x0r = x1r + x3i
		x0i = x1i - x3r
		d[14] = float32(wk3s[0]*x0r) - float32(wk3s[1]*x0i)
		d[15] = float32(wk3s[0]*x0i) + float32(wk3s[1]*x0r)
	}
}

// cftmdlFastScalar is the scalar cftmdl_128 from s into a.
func cftmdlFastScalar(s, a *[128]float32) {
	wk1r := rdftW[2]
	for j := 0; j < 8; j += 2 {
		p0, p1, p2, p3 := (*[8]float32)(s[0:8]), (*[8]float32)(s[8:16]), (*[8]float32)(s[16:24]), (*[8]float32)(s[24:32])
		x0r := p0[j] + p1[j]
		x0i := p0[j+1] + p1[j+1]
		x1r := p0[j] - p1[j]
		x1i := p0[j+1] - p1[j+1]
		x2r := p2[j] + p3[j]
		x2i := p2[j+1] + p3[j+1]
		x3r := p2[j] - p3[j]
		x3i := p2[j+1] - p3[j+1]
		d0, d1, d2, d3 := (*[8]float32)(a[0:8]), (*[8]float32)(a[8:16]), (*[8]float32)(a[16:24]), (*[8]float32)(a[24:32])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		d2[j] = x0r - x2r
		d2[j+1] = x0i - x2i
		d1[j] = x1r - x3i
		d1[j+1] = x1i + x3r
		d3[j] = x1r + x3i
		d3[j+1] = x1i - x3r

		p0, p1, p2, p3 = (*[8]float32)(s[32:40]), (*[8]float32)(s[40:48]), (*[8]float32)(s[48:56]), (*[8]float32)(s[56:64])
		x0r = p0[j] + p1[j]
		x0i = p0[j+1] + p1[j+1]
		x1r = p0[j] - p1[j]
		x1i = p0[j+1] - p1[j+1]
		x2r = p2[j] + p3[j]
		x2i = p2[j+1] + p3[j+1]
		x3r = p2[j] - p3[j]
		x3i = p2[j+1] - p3[j+1]
		d0, d1, d2, d3 = (*[8]float32)(a[32:40]), (*[8]float32)(a[40:48]), (*[8]float32)(a[48:56]), (*[8]float32)(a[56:64])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		d2[j] = x2i - x0i
		d2[j+1] = x0r - x2r
		x0r = x1r - x3i
		x0i = x1i + x3r
		d1[j] = wk1r * (x0r - x0i)
		d1[j+1] = wk1r * (x0r + x0i)
		x0r = x3i + x1r
		x0i = x3r - x1i
		d3[j] = wk1r * (x0i - x0r)
		d3[j+1] = wk1r * (x0i + x0r)
	}
	// k = 64: k1 = 2, k2 = 4.
	wk2r, wk2i := rdftW[2], rdftW[3]
	wk1r, wk1i := rdftW[4], rdftW[5]
	wk3r, wk3i := rdftWk3riFirst[2], rdftWk3riFirst[3]
	vk1r, vk1i := rdftW[6], rdftW[7]
	vk3r, vk3i := rdftWk3riSecond[2], rdftWk3riSecond[3]
	for j := 0; j < 8; j += 2 {
		p0, p1, p2, p3 := (*[8]float32)(s[64:72]), (*[8]float32)(s[72:80]), (*[8]float32)(s[80:88]), (*[8]float32)(s[88:96])
		x0r := p0[j] + p1[j]
		x0i := p0[j+1] + p1[j+1]
		x1r := p0[j] - p1[j]
		x1i := p0[j+1] - p1[j+1]
		x2r := p2[j] + p3[j]
		x2i := p2[j+1] + p3[j+1]
		x3r := p2[j] - p3[j]
		x3i := p2[j+1] - p3[j+1]
		d0, d1, d2, d3 := (*[8]float32)(a[64:72]), (*[8]float32)(a[72:80]), (*[8]float32)(a[80:88]), (*[8]float32)(a[88:96])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		x0r -= x2r
		x0i -= x2i
		d2[j] = float32(wk2r*x0r) - float32(wk2i*x0i)
		d2[j+1] = float32(wk2r*x0i) + float32(wk2i*x0r)
		x0r = x1r - x3i
		x0i = x1i + x3r
		d1[j] = float32(wk1r*x0r) - float32(wk1i*x0i)
		d1[j+1] = float32(wk1r*x0i) + float32(wk1i*x0r)
		x0r = x1r + x3i
		x0i = x1i - x3r
		d3[j] = float32(wk3r*x0r) - float32(wk3i*x0i)
		d3[j+1] = float32(wk3r*x0i) + float32(wk3i*x0r)

		p0, p1, p2, p3 = (*[8]float32)(s[96:104]), (*[8]float32)(s[104:112]), (*[8]float32)(s[112:120]), (*[8]float32)(s[120:128])
		x0r = p0[j] + p1[j]
		x0i = p0[j+1] + p1[j+1]
		x1r = p0[j] - p1[j]
		x1i = p0[j+1] - p1[j+1]
		x2r = p2[j] + p3[j]
		x2i = p2[j+1] + p3[j+1]
		x3r = p2[j] - p3[j]
		x3i = p2[j+1] - p3[j+1]
		d0, d1, d2, d3 = (*[8]float32)(a[96:104]), (*[8]float32)(a[104:112]), (*[8]float32)(a[112:120]), (*[8]float32)(a[120:128])
		d0[j] = x0r + x2r
		d0[j+1] = x0i + x2i
		x0r -= x2r
		x0i -= x2i
		d2[j] = float32(-wk2i*x0r) - float32(wk2r*x0i)
		d2[j+1] = float32(-wk2i*x0i) + float32(wk2r*x0r)
		x0r = x1r - x3i
		x0i = x1i + x3r
		d1[j] = float32(vk1r*x0r) - float32(vk1i*x0i)
		d1[j+1] = float32(vk1r*x0i) + float32(vk1i*x0r)
		x0r = x1r + x3i
		x0i = x1i - x3r
		d3[j] = float32(vk3r*x0r) - float32(vk3i*x0i)
		d3[j+1] = float32(vk3r*x0i) + float32(vk3i*x0r)
	}
}

// cftbsubTailFast is cftbsubTail without bounds checks.
func cftbsubTailFast(a *[128]float32) {
	p0, p1, p2, p3 := (*[32]float32)(a[0:32]), (*[32]float32)(a[32:64]), (*[32]float32)(a[64:96]), (*[32]float32)(a[96:128])
	for j := 0; j < 32; j += 2 {
		x0r := p0[j] + p1[j]
		x0i := -p0[j+1] - p1[j+1]
		x1r := p0[j] - p1[j]
		x1i := -p0[j+1] + p1[j+1]
		x2r := p2[j] + p3[j]
		x2i := p2[j+1] + p3[j+1]
		x3r := p2[j] - p3[j]
		x3i := p2[j+1] - p3[j+1]
		p0[j] = x0r + x2r
		p0[j+1] = x0i - x2i
		p2[j] = x0r - x2r
		p2[j+1] = x0i + x2i
		p1[j] = x1r - x3i
		p1[j+1] = x1i - x3r
		p3[j] = x1r + x3i
		p3[j+1] = x1i + x3r
	}
}

// rftbsubFast is rftbsub_128, laid out like rftfsubFast.
func rftbsubFast(a *[128]float32) {
	a[1] = -a[1]
	lo := (*[62]float32)(a[2:64])
	hi := (*[62]float32)(a[66:128])
	c := (*[31]float32)(rdftW[33:64])
	for m := 0; m < 31; m++ {
		n := 30 - m
		wkr := 0.5 - c[n]
		wki := c[m]
		l := (*[2]float32)(lo[2*m : 2*m+2])
		h := (*[2]float32)(hi[2*n : 2*n+2])
		xr := l[0] - h[0]
		xi := l[1] + h[1]
		yr := float32(wkr*xr) + float32(wki*xi)
		yi := float32(wkr*xi) - float32(wki*xr)
		l[0] = l[0] - yr
		l[1] = yi - l[1]
		h[0] = yr + h[0]
		h[1] = yi - h[1]
	}
	a[65] = -a[65]
}

// Split transforms.
//
// FftSplit ends with cftfsubTail, rftfsub and the copy to bins, and
// InverseFftSplit starts with the copy from bins, rftbsub and (after
// bitrv2) cft1st. Run as passes, each loads and stores the whole frame. The
// functions below run each group of three as one pass that loads every
// input once and stores every output once, with the float operations of the
// passes per element.
//
// rftfsub and rftbsub work on the bin pairs (k, 64-k), k = 1..31. The last
// radix-4 butterfly of the forward transform at index c produces the bins c,
// 16+c, 32+c and 48+c, and the first butterfly of the inverse (after
// bitrv2) at index c consumes the same four. The partners of these bins are
// the bins of the butterfly at 16-c, so one step runs the butterflies c and
// 16-c (c = 1..7) with the four pair operations between them; the
// butterflies 0 and 8 hold their own partners (16 with 48; 8 with 56 and 24
// with 40), and the bins 0 and 32 have none.

// rftWk[c] holds the rftfsub/rftbsub factors (wkr, wki) of the bin pairs at
// k = c, 32-c, 16-c and 16+c: wkr = 0.5 - w[64-k] and wki = w[32+k] with
// w = rdftW, as the passes compute them.
var rftWk = makeRftWk()

func makeRftWk() (t [9][8]float32) {
	for c := range t {
		for i, k := range [4]int{c, 32 - c, 16 - c, 16 + c} {
			if k < 1 || k > 31 {
				continue
			}
			t[c][2*i] = 0.5 - rdftW[64-k]
			t[c][2*i+1] = rdftW[32+k]
		}
	}
	return t
}

// rftfPair is one iteration of rftfsub_128 on the bins l = k and h = 64-k.
func rftfPair(wkr, wki, lr, li, hr, hi float32) (float32, float32, float32, float32) {
	xr := lr - hr
	xi := li + hi
	yr := float32(wkr*xr) - float32(wki*xi)
	yi := float32(wkr*xi) + float32(wki*xr)
	return lr - yr, li - yi, hr + yr, hi - yi
}

// rftbPair is one iteration of rftbsub_128 on the bins l = k and h = 64-k.
func rftbPair(wkr, wki, lr, li, hr, hi float32) (float32, float32, float32, float32) {
	xr := lr - hr
	xi := li + hi
	yr := float32(wkr*xr) + float32(wki*xi)
	yi := float32(wkr*xi) - float32(wki*xr)
	return lr - yr, yi - li, yr + hr, yi - hi
}

func fftSplitFastLane(a *[128]float32, re, im *[65]float32) {
	var t [128]float32
	cft1stFastLane(a, &t)
	cftmdlFastLane(&t, a)
	tailSplitFast(a, re, im)
}

func fftSplitFastScalar(a *[128]float32, re, im *[65]float32) {
	var t [128]float32
	cft1stFastScalar(a, &t)
	cftmdlFastScalar(&t, a)
	tailSplitFast(a, re, im)
}

func inverseFftSplitFastLane(re, im *[65]float32, a *[128]float32) {
	var t [128]float32
	splitCft1stFastLane(re, im, &t)
	cftmdlFastLane(&t, a)
	cftbsubTailFast(a)
}

func inverseFftSplitFastScalar(re, im *[65]float32, a *[128]float32) {
	pack(re, im, a)
	inverseFftFastScalar(a)
}

// tailSplitFast is cftfsubTail, rftfsub, the exchange of a[0] and a[1] that
// ends the forward transform and FftData::CopyFromPackedArray, from a into
// re and im.
//
// A step computes its eight bins in two halves, each from the frame: the
// first half takes the sums of butterfly c and the differences of butterfly
// 16-c (bins c, 32+c, 32-c, 64-c), the second half the rest. A half holds
// eight values at a time, which fits the float registers of amd64, where
// the second operand of every sum and difference is read from memory.
func tailSplitFast(a *[128]float32, re, im *[65]float32) {
	// Butterfly 0: bins 0, 16, 32, 48.
	{
		x0r := a[0] + a[32]
		x0i := a[1] + a[33]
		x1r := a[0] - a[32]
		x1i := a[1] - a[33]
		x2r := a[64] + a[96]
		x2i := a[65] + a[97]
		x3r := a[64] - a[96]
		x3i := a[65] - a[97]
		b0r := x0r + x2r
		b0i := x0i + x2i
		re[0] = b0r + b0i
		re[64] = b0r - b0i
		im[0] = 0
		im[64] = 0
		re[32] = x0r - x2r
		im[32] = x0i - x2i
		w := &rftWk[0]
		lr, li, hr, hi := rftfPair(w[6], w[7], x1r-x3i, x1i+x3r, x1r+x3i, x1i-x3r)
		re[16], im[16], re[48], im[48] = lr, li, hr, hi
	}
	// Butterfly 8: bins 8, 24, 40, 56.
	{
		x0r := a[16] + a[48]
		x0i := a[17] + a[49]
		x1r := a[16] - a[48]
		x1i := a[17] - a[49]
		x2r := a[80] + a[112]
		x2i := a[81] + a[113]
		x3r := a[80] - a[112]
		x3i := a[81] - a[113]
		w := &rftWk[8]
		lr, li, hr, hi := rftfPair(w[0], w[1], x0r+x2r, x0i+x2i, x1r+x3i, x1i-x3r)
		re[8], im[8], re[56], im[56] = lr, li, hr, hi
		lr, li, hr, hi = rftfPair(w[2], w[3], x1r-x3i, x1i+x3r, x0r-x2r, x0i-x2i)
		re[24], im[24], re[40], im[40] = lr, li, hr, hi
	}
	for c := 1; c < 8; c++ {
		// p[0], p[32], p[64], p[96] are the inputs of butterfly c and q[...]
		// those of butterfly 16-c; the outputs go to bins c + {0, 16, 32, 48}
		// and 16-c + {0, 16, 32, 48}.
		n := (16 - c) & 15
		p := (*[98]float32)(a[2*c : 2*c+98])
		q := (*[98]float32)(a[2*n : 2*n+98])
		rp, ip := (*[49]float32)(re[c:c+49]), (*[49]float32)(im[c:c+49])
		rq, iq := (*[49]float32)(re[n:n+49]), (*[49]float32)(im[n:n+49])
		w := &rftWk[c]

		// Sums of c, differences of 16-c.
		x0r := p[0] + p[32]
		x0i := p[1] + p[33]
		x2r := p[64] + p[96]
		x2i := p[65] + p[97]
		y1r := q[0] - q[32]
		y1i := q[1] - q[33]
		y3r := q[64] - q[96]
		y3i := q[65] - q[97]
		// Bins c and 64-c.
		lr, li, hr, hi := rftfPair(w[0], w[1], x0r+x2r, x0i+x2i, y1r+y3i, y1i-y3r)
		rp[0], ip[0], rq[48], iq[48] = lr, li, hr, hi
		// Bins 32-c and 32+c.
		lr, li, hr, hi = rftfPair(w[2], w[3], y1r-y3i, y1i+y3r, x0r-x2r, x0i-x2i)
		rq[16], iq[16], rp[32], ip[32] = lr, li, hr, hi

		// Differences of c, sums of 16-c.
		x1r := p[0] - p[32]
		x1i := p[1] - p[33]
		x3r := p[64] - p[96]
		x3i := p[65] - p[97]
		y0r := q[0] + q[32]
		y0i := q[1] + q[33]
		y2r := q[64] + q[96]
		y2i := q[65] + q[97]
		// Bins 16-c and 48+c.
		lr, li, hr, hi = rftfPair(w[4], w[5], y0r+y2r, y0i+y2i, x1r+x3i, x1i-x3r)
		rq[0], iq[0], rp[48], ip[48] = lr, li, hr, hi
		// Bins 16+c and 48-c.
		lr, li, hr, hi = rftfPair(w[6], w[7], x1r-x3i, x1i+x3r, y0r-y2r, y0i-y2i)
		rp[16], ip[16], rq[32], iq[32] = lr, li, hr, hi
	}
}

// splitCft1stFastLane is FftData::CopyToPackedArray, the first two
// statements of the inverse transform, rftbsub, bitrv2 and the NEON/SSE2
// cft1st, from re and im into t. It reads neither im[0] nor im[64].
//
// Group s of cft1st (see cft1stFastLane) reads the bins s, s+32, s+16 and
// s+48 in this order (c0..c3 below) and writes t[16q+8h:] with
// q = rev3(s&7) and h = s>>3, with the twiddle factors at 4q+2h.
func splitCft1stFastLane(re, im *[65]float32, t *[128]float32) {
	// Group 0: bins 0, 32, 16, 48.
	{
		a1 := float32(0.5 * (re[0] - re[64]))
		c0r, c0i := re[0]-a1, -a1
		c1r, c1i := re[32], -im[32]
		w := &rftWk[0]
		c2r, c2i, c3r, c3i := rftbPair(w[6], w[7], re[16], im[16], re[48], im[48])
		{
			d, k := (*[8]float32)(t[0:8]), 0
			w1r, w1i := (*[2]float32)(rdftWk1r[k:k+2]), (*[2]float32)(rdftWk1i[k:k+2])
			w2r, w2i := (*[2]float32)(rdftWk2r[k:k+2]), (*[2]float32)(rdftWk2i[k:k+2])
			w3r, w3i := (*[2]float32)(rdftWk3r[k:k+2]), (*[2]float32)(rdftWk3i[k:k+2])
			x0r := c0r + c1r
			x0i := c0i + c1i
			x1r := c0r - c1r
			x1i := c0i - c1i
			x2r := c2r + c3r
			x2i := c2i + c3i
			x3r := c2r - c3r
			x3i := c2i - c3i
			d[0] = x0r + x2r
			d[1] = x0i + x2i
			yr := x0r - x2r
			yi := x0i - x2i
			d[4] = float32(w2r[0]*yr) + float32(w2i[0]*yi)
			d[5] = float32(w2r[1]*yi) + float32(w2i[1]*yr)
			yr = x1r - x3i
			yi = x1i + x3r
			d[2] = float32(w1r[0]*yr) + float32(w1i[0]*yi)
			d[3] = float32(w1r[1]*yi) + float32(w1i[1]*yr)
			yr = x1r + x3i
			yi = x1i - x3r
			d[6] = float32(w3r[0]*yr) + float32(w3i[0]*yi)
			d[7] = float32(w3r[1]*yi) + float32(w3i[1]*yr)
		}
	}
	// Group 8: bins 8, 40, 24, 56.
	{
		w := &rftWk[8]
		c0r, c0i, c3r, c3i := rftbPair(w[0], w[1], re[8], im[8], re[56], im[56])
		c2r, c2i, c1r, c1i := rftbPair(w[2], w[3], re[24], im[24], re[40], im[40])
		{
			d, k := (*[8]float32)(t[8:16]), 2
			w1r, w1i := (*[2]float32)(rdftWk1r[k:k+2]), (*[2]float32)(rdftWk1i[k:k+2])
			w2r, w2i := (*[2]float32)(rdftWk2r[k:k+2]), (*[2]float32)(rdftWk2i[k:k+2])
			w3r, w3i := (*[2]float32)(rdftWk3r[k:k+2]), (*[2]float32)(rdftWk3i[k:k+2])
			x0r := c0r + c1r
			x0i := c0i + c1i
			x1r := c0r - c1r
			x1i := c0i - c1i
			x2r := c2r + c3r
			x2i := c2i + c3i
			x3r := c2r - c3r
			x3i := c2i - c3i
			d[0] = x0r + x2r
			d[1] = x0i + x2i
			yr := x0r - x2r
			yi := x0i - x2i
			d[4] = float32(w2r[0]*yr) + float32(w2i[0]*yi)
			d[5] = float32(w2r[1]*yi) + float32(w2i[1]*yr)
			yr = x1r - x3i
			yi = x1i + x3r
			d[2] = float32(w1r[0]*yr) + float32(w1i[0]*yi)
			d[3] = float32(w1r[1]*yi) + float32(w1i[1]*yr)
			yr = x1r + x3i
			yi = x1i - x3r
			d[6] = float32(w3r[0]*yr) + float32(w3i[0]*yi)
			d[7] = float32(w3r[1]*yi) + float32(w3i[1]*yr)
		}
	}
	for s := 1; s < 8; s++ {
		// Group s reads the bins s + {0, 32, 16, 48} (p0..p3) and group
		// 16-s the bins 16-s + {0, 32, 16, 48} (q0..q3).
		n := (16 - s) & 15
		rp, ip := (*[49]float32)(re[s:s+49]), (*[49]float32)(im[s:s+49])
		rq, iq := (*[49]float32)(re[n:n+49]), (*[49]float32)(im[n:n+49])
		w := &rftWk[s]
		// Bins s and 64-s.
		p0r, p0i, q3r, q3i := rftbPair(w[0], w[1], rp[0], ip[0], rq[48], iq[48])
		// Bins 32-s and 32+s.
		q2r, q2i, p1r, p1i := rftbPair(w[2], w[3], rq[16], iq[16], rp[32], ip[32])
		// Bins 16-s and 48+s.
		q0r, q0i, p3r, p3i := rftbPair(w[4], w[5], rq[0], iq[0], rp[48], ip[48])
		// Bins 16+s and 48-s.
		p2r, p2i, q1r, q1i := rftbPair(w[6], w[7], rp[16], ip[16], rq[32], iq[32])

		qs := int(rev3[s]) & 7
		qn := int(rev3[n&7]) & 7
		// The masks change no value; they bound the windows for the
		// compiler.
		os, on := 16*qs, (16*qn+8)&120
		ks, kn := 4*qs, (4*qn+2)&30
		{
			d, k := (*[8]float32)(t[os:os+8]), ks
			w1r, w1i := (*[2]float32)(rdftWk1r[k:k+2]), (*[2]float32)(rdftWk1i[k:k+2])
			w2r, w2i := (*[2]float32)(rdftWk2r[k:k+2]), (*[2]float32)(rdftWk2i[k:k+2])
			w3r, w3i := (*[2]float32)(rdftWk3r[k:k+2]), (*[2]float32)(rdftWk3i[k:k+2])
			x0r := p0r + p1r
			x0i := p0i + p1i
			x1r := p0r - p1r
			x1i := p0i - p1i
			x2r := p2r + p3r
			x2i := p2i + p3i
			x3r := p2r - p3r
			x3i := p2i - p3i
			d[0] = x0r + x2r
			d[1] = x0i + x2i
			yr := x0r - x2r
			yi := x0i - x2i
			d[4] = float32(w2r[0]*yr) + float32(w2i[0]*yi)
			d[5] = float32(w2r[1]*yi) + float32(w2i[1]*yr)
			yr = x1r - x3i
			yi = x1i + x3r
			d[2] = float32(w1r[0]*yr) + float32(w1i[0]*yi)
			d[3] = float32(w1r[1]*yi) + float32(w1i[1]*yr)
			yr = x1r + x3i
			yi = x1i - x3r
			d[6] = float32(w3r[0]*yr) + float32(w3i[0]*yi)
			d[7] = float32(w3r[1]*yi) + float32(w3i[1]*yr)
		}
		{
			d, k := (*[8]float32)(t[on:on+8]), kn
			w1r, w1i := (*[2]float32)(rdftWk1r[k:k+2]), (*[2]float32)(rdftWk1i[k:k+2])
			w2r, w2i := (*[2]float32)(rdftWk2r[k:k+2]), (*[2]float32)(rdftWk2i[k:k+2])
			w3r, w3i := (*[2]float32)(rdftWk3r[k:k+2]), (*[2]float32)(rdftWk3i[k:k+2])
			x0r := q0r + q1r
			x0i := q0i + q1i
			x1r := q0r - q1r
			x1i := q0i - q1i
			x2r := q2r + q3r
			x2i := q2i + q3i
			x3r := q2r - q3r
			x3i := q2i - q3i
			d[0] = x0r + x2r
			d[1] = x0i + x2i
			yr := x0r - x2r
			yi := x0i - x2i
			d[4] = float32(w2r[0]*yr) + float32(w2i[0]*yi)
			d[5] = float32(w2r[1]*yi) + float32(w2i[1]*yr)
			yr = x1r - x3i
			yi = x1i + x3r
			d[2] = float32(w1r[0]*yr) + float32(w1i[0]*yi)
			d[3] = float32(w1r[1]*yi) + float32(w1i[1]*yr)
			yr = x1r + x3i
			yi = x1i - x3r
			d[6] = float32(w3r[0]*yr) + float32(w3i[0]*yi)
			d[7] = float32(w3r[1]*yi) + float32(w3i[1]*yr)
		}
	}
}
