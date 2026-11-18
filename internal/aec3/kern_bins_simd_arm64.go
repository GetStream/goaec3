//go:build goexperiment.simd && arm64 && !purego

package aec3

import "simd/archsimd"

// archsimd (NEON) kernels for the kern_bins.go helpers.

type f32x4 = archsimd.Float32x4

func kernScale(x []float32, a float32, z []float32) {
	z = z[:len(x)]
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
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[8]float32)(v[2*k : 2*k+8])
		a := archsimd.LoadFloat32x4Array((*[4]float32)(p[0:4])).ToBits()
		b := archsimd.LoadFloat32x4Array((*[4]float32)(p[4:8])).ToBits()
		a.ConcatEven(b).BitsToFloat32().StoreArray((*[4]float32)(re[k : k+4]))
		a.ConcatOdd(b).BitsToFloat32().StoreArray((*[4]float32)(im[k : k+4]))
	}
	re[FftLengthBy2] = im[0]
	im[0] = 0
	im[FftLengthBy2] = 0
}

// kernPack interleaves re[0..63] and im[0..63] into v, then puts the Nyquist
// real part re[64] in v[1].
func kernPack(re, im *[FftLengthBy2Plus1]float32, v *[FftLength]float32) {
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
		vmn := archsimd.LoadFloat32x4(x)
		vmx := vmn
		j := 4
		for ; j+4 <= len(x); j += 4 {
			v := archsimd.LoadFloat32x4(x[j:])
			vmn = vmn.Min(v)
			vmx = vmx.Max(v)
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

// vStdMin is std::min(a, b) per lane: b < a ? b : a. FMIN differs from it
// on NaN and on equal zeros, so it is a compare and a select.
func vStdMin(a, b f32x4) f32x4 { return b.IfElse(b.Less(a), a) }

// vStdMax is std::max(a, b) per lane: a < b ? b : a.
func vStdMax(a, b f32x4) f32x4 { return b.IfElse(a.Less(b), a) }

// kernClamp is kernClampGeneric: std::min(std::max(x, lo), hi).
func kernClamp(x []float32, lo, hi float32) {
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

func kernMulAdd2(a, b, c, d, z []float32) {
	n := len(z)
	a, b, c, d = a[:n], b[:n], c[:n], d[:n]
	j := 0
	for ; j+4 <= n; j += 4 {
		p := archsimd.LoadFloat32x4(a[j:]).Mul(archsimd.LoadFloat32x4(b[j:]))
		q := archsimd.LoadFloat32x4(c[j:]).Mul(archsimd.LoadFloat32x4(d[j:]))
		p.Add(q).Store(z[j:])
	}
	for ; j < n; j++ {
		z[j] = float32(a[j]*b[j]) + float32(c[j]*d[j])
	}
}

func kernAxpy(x []float32, a float32, z []float32) {
	z = z[:len(x)]
	av := archsimd.BroadcastFloat32x4(a)
	j := 0
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(z[j:]).Add(archsimd.LoadFloat32x4(x[j:]).Mul(av)).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] += float32(x[j] * a)
	}
}

func kernOneMinusSq(x, z []float32) {
	z = z[:len(x)]
	one := archsimd.BroadcastFloat32x4(1)
	j := 0
	for ; j+4 <= len(x); j += 4 {
		v := archsimd.LoadFloat32x4(x[j:])
		one.Sub(v.Mul(v)).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = 1 - float32(x[j]*x[j])
	}
}

func kernDiffPower(yr, er, yi, ei, z []float32) {
	n := len(z)
	yr, er, yi, ei = yr[:n], er[:n], yi[:n], ei[:n]
	j := 0
	for ; j+4 <= n; j += 4 {
		dr := archsimd.LoadFloat32x4(yr[j:]).Sub(archsimd.LoadFloat32x4(er[j:]))
		di := archsimd.LoadFloat32x4(yi[j:]).Sub(archsimd.LoadFloat32x4(ei[j:]))
		dr.Mul(dr).Add(di.Mul(di)).Store(z[j:])
	}
	for ; j < n; j++ {
		dr := yr[j] - er[j]
		di := yi[j] - ei[j]
		z[j] = float32(dr*dr) + float32(di*di)
	}
}

func kernStdMin(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	j := 0
	for ; j+4 <= len(x); j += 4 {
		vStdMin(archsimd.LoadFloat32x4(x[j:]), archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = stdMin(x[j], y[j])
	}
}

func kernStdMaxScalar(x []float32, a float32) {
	av := archsimd.BroadcastFloat32x4(a)
	j := 0
	for ; j+4 <= len(x); j += 4 {
		vStdMax(archsimd.LoadFloat32x4(x[j:]), av).Store(x[j:])
	}
	for ; j < len(x); j++ {
		x[j] = stdMax(x[j], a)
	}
}

func kernSmooth(s, y []float32, c float32) {
	y = y[:len(s)]
	cv := archsimd.BroadcastFloat32x4(c)
	j := 0
	for ; j+4 <= len(s); j += 4 {
		a := archsimd.LoadFloat32x4(s[j:])
		a.Add(cv.Mul(archsimd.LoadFloat32x4(y[j:]).Sub(a))).Store(s[j:])
	}
	for ; j < len(s); j++ {
		a := s[j]
		s[j] = a + float32(c*(y[j]-a))
	}
}

// kernNoAudibleEcho evaluates both branches per lane and selects; Div is
// IEEE division.
func kernNoAudibleEcho(nearend, echo, masker, enrT, enrS, emrT, gain *[FftLengthBy2Plus1]float32) {
	one := archsimd.BroadcastFloat32x4(1)
	for k := 0; k < FftLengthBy2; k += 4 {
		e := archsimd.LoadFloat32x4Array((*[4]float32)(echo[k : k+4]))
		enr := e.Div(archsimd.LoadFloat32x4Array((*[4]float32)(nearend[k : k+4])).Add(one))
		emr := e.Div(archsimd.LoadFloat32x4Array((*[4]float32)(masker[k : k+4])).Add(one))
		et := archsimd.LoadFloat32x4Array((*[4]float32)(enrT[k : k+4]))
		es := archsimd.LoadFloat32x4Array((*[4]float32)(enrS[k : k+4]))
		mt := archsimd.LoadFloat32x4Array((*[4]float32)(emrT[k : k+4]))
		g := vStdMax(es.Sub(enr).Div(es.Sub(et)), mt.Div(emr))
		g.IfElse(enr.Greater(et).And(emr.Greater(mt)), one).StoreArray((*[4]float32)(gain[k : k+4]))
	}
	const n = FftLengthBy2
	kernNoAudibleEchoBin(nearend, echo, masker, enrT, enrS, emrT, gain, n)
}

func kernNoAudibleEchoBin(nearend, echo, masker, enrT, enrS, emrT, gain *[FftLengthBy2Plus1]float32, k int) {
	enr := echo[k] / (nearend[k] + 1)
	emr := echo[k] / (masker[k] + 1)
	g := float32(1)
	if enr > enrT[k] && emr > emrT[k] {
		g = (enrS[k] - enr) / (enrS[k] - enrT[k])
		g = stdMax(g, emrT[k]/emr)
	}
	gain[k] = g
}

func kernMinGain(w *[FftLengthBy2Plus1]float32, minEchoPower float32, minGain *[FftLengthBy2Plus1]float32) {
	one := archsimd.BroadcastFloat32x4(1)
	var zero f32x4
	mp := archsimd.BroadcastFloat32x4(minEchoPower)
	for k := 0; k < FftLengthBy2; k += 4 {
		wv := archsimd.LoadFloat32x4Array((*[4]float32)(w[k : k+4]))
		vStdMin(mp.Div(wv).IfElse(wv.Greater(zero), one), one).StoreArray((*[4]float32)(minGain[k : k+4]))
	}
	const n = FftLengthBy2
	if w[n] > 0 {
		minGain[n] = minEchoPower / w[n]
	} else {
		minGain[n] = 1
	}
	minGain[n] = stdMin(minGain[n], 1)
}

func kernWeigh(threshold, normalizer float32, echo, weightedEcho []float32) {
	weightedEcho = weightedEcho[:len(echo)]
	th := archsimd.BroadcastFloat32x4(threshold)
	nv := archsimd.BroadcastFloat32x4(normalizer)
	one := archsimd.BroadcastFloat32x4(1)
	var zero f32x4
	j := 0
	for ; j+4 <= len(echo); j += 4 {
		e := archsimd.LoadFloat32x4(echo[j:])
		tmp := th.Sub(e).Mul(nv)
		w := e.Mul(vStdMax(zero, one.Sub(tmp.Mul(tmp))))
		w.IfElse(e.Less(th), e).Store(weightedEcho[j:])
	}
	kernWeighGeneric(threshold, normalizer, echo[j:], weightedEcho[j:])
}

func kernNoiseUpdate(n, s *[FftLengthBy2Plus1]float32) {
	c09 := archsimd.BroadcastFloat32x4(0.9)
	c01 := archsimd.BroadcastFloat32x4(0.1)
	c1 := archsimd.BroadcastFloat32x4(1.0002)
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[4]float32)(n[k : k+4])
		a := archsimd.LoadFloat32x4Array(p)
		b := archsimd.LoadFloat32x4Array((*[4]float32)(s[k : k+4]))
		u := c09.Mul(b).Add(c01.Mul(a)).Mul(c1)
		u.IfElse(b.Less(a), a.Mul(c1)).StoreArray(p)
	}
	kernNoiseUpdateBin(n, s, FftLengthBy2)
}

func kernNoiseUpdateBin(n, s *[FftLengthBy2Plus1]float32, k int) {
	a, b := n[k], s[k]
	if b < a {
		n[k] = (float32(0.9*b) + float32(0.1*a)) * 1.0002
	} else {
		n[k] = a * 1.0002
	}
}

func kernNoiseInitialUpdate(n, ini *[FftLengthBy2Plus1]float32) {
	c := archsimd.BroadcastFloat32x4(0.001)
	for k := 0; k < FftLengthBy2; k += 4 {
		p := (*[4]float32)(ini[k : k+4])
		a := archsimd.LoadFloat32x4Array((*[4]float32)(n[k : k+4]))
		b := archsimd.LoadFloat32x4Array(p)
		b.Add(c.Mul(a.Sub(b))).IfElse(a.Greater(b), a).StoreArray(p)
	}
	const k = FftLengthBy2
	a, b := n[k], ini[k]
	if a > b {
		ini[k] = b + float32(0.001*(a-b))
	} else {
		ini[k] = a
	}
}

func kernFir3(x []float32, h *[3]float32, out []float32, from, to int) {
	h0 := archsimd.BroadcastFloat32x4(h[0])
	h1 := archsimd.BroadcastFloat32x4(h[1])
	h2 := archsimd.BroadcastFloat32x4(h[2])
	var zero f32x4
	k := from
	if from >= 2 {
		for ; k+3 <= to && k+4 <= len(x) && k+4 <= len(out); k += 4 {
			t := zero.Add(archsimd.LoadFloat32x4(x[k:]).Mul(h0))
			t = t.Add(archsimd.LoadFloat32x4(x[k-1:]).Mul(h1))
			t = t.Add(archsimd.LoadFloat32x4(x[k-2:]).Mul(h2))
			t.Store(out[k:])
		}
	}
	kernFir3Generic(x, h, out, k, to)
}

func kernDiv(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	j := 0
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(x[j:]).Div(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = x[j] / y[j]
	}
}

func kernReverb(r, p, s []float32, d float32) {
	p = p[:len(r)]
	s = s[:len(r)]
	dv := archsimd.BroadcastFloat32x4(d)
	j := 0
	for ; j+4 <= len(r); j += 4 {
		q := archsimd.LoadFloat32x4(p[j:]).Mul(archsimd.LoadFloat32x4(s[j:]))
		archsimd.LoadFloat32x4(r[j:]).Add(q).Mul(dv).Store(r[j:])
	}
	for ; j < len(r); j++ {
		r[j] = (r[j] + float32(p[j]*s[j])) * d
	}
}

// kernStdMax selects with a compare, exactly like std::max.
func kernStdMax(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	j := 0
	for ; j+4 <= len(x); j += 4 {
		vStdMax(archsimd.LoadFloat32x4(x[j:]), archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = stdMax(x[j], y[j])
	}
}

func kernLd4(a *[FftLengthBy2Plus1]float32, k int) f32x4 {
	return archsimd.LoadFloat32x4Array((*[4]float32)(a[k : k+4]))
}

func kernSt4(a *[FftLengthBy2Plus1]float32, k int, v f32x4) {
	v.StoreArray((*[4]float32)(a[k : k+4]))
}

func kernRefinedMu(X2, hError, E2 *[FftLengthBy2Plus1]float32, gate, sp float32, mu *[FftLengthBy2Plus1]float32) {
	half := archsimd.BroadcastFloat32x4(0.5)
	gv := archsimd.BroadcastFloat32x4(gate)
	spv := archsimd.BroadcastFloat32x4(sp)
	var zero f32x4
	for k := 0; k < FftLengthBy2; k += 4 {
		x := kernLd4(X2, k)
		h := kernLd4(hError, k)
		d := half.Mul(h).Mul(x).Add(spv.Mul(kernLd4(E2, k)))
		kernSt4(mu, k, h.Div(d).IfElse(x.GreaterEqual(gv), zero))
	}
	const k = FftLengthBy2
	if X2[k] >= gate {
		h := hError[k]
		mu[k] = h / (float32(float32(0.5*h)*X2[k]) + float32(sp*E2[k]))
	} else {
		mu[k] = 0
	}
}

func kernRefinedHError(mu, X2, hError *[FftLengthBy2Plus1]float32) {
	half := archsimd.BroadcastFloat32x4(0.5)
	for k := 0; k < FftLengthBy2; k += 4 {
		h := kernLd4(hError, k)
		kernSt4(hError, k, h.Sub(half.Mul(kernLd4(mu, k)).Mul(kernLd4(X2, k)).Mul(h)))
	}
	const k = FftLengthBy2
	hError[k] -= float32(float32(float32(0.5*mu[k])*X2[k]) * hError[k])
}

// kernRefinedLeak selects the leakage per lane.
func kernRefinedLeak(E2Refined, E2Coarse *[FftLengthBy2Plus1]float32, erl []float32, disallowLeakageDiverged bool,
	leakageConverged, leakageDiverged, errorFloor, errorCeil float32, hError *[FftLengthBy2Plus1]float32) {
	e := (*[FftLengthBy2Plus1]float32)(erl)
	lc := archsimd.BroadcastFloat32x4(leakageConverged)
	ld := archsimd.BroadcastFloat32x4(leakageDiverged)
	fl := archsimd.BroadcastFloat32x4(errorFloor)
	ce := archsimd.BroadcastFloat32x4(errorCeil)
	for k := 0; k < FftLengthBy2; k += 4 {
		l := lc
		if !disallowLeakageDiverged {
			l = lc.IfElse(kernLd4(E2Refined, k).LessEqual(kernLd4(E2Coarse, k)), ld)
		}
		h := kernLd4(hError, k).Add(l.Mul(kernLd4(e, k)))
		kernSt4(hError, k, vStdMin(vStdMax(h, fl), ce))
	}
	const k = FftLengthBy2
	if E2Refined[k] <= E2Coarse[k] || disallowLeakageDiverged {
		hError[k] += float32(leakageConverged * e[k])
	} else {
		hError[k] += float32(leakageDiverged * e[k])
	}
	hError[k] = stdMax(hError[k], errorFloor)
	hError[k] = stdMin(hError[k], errorCeil)
}

func kernCoarseMu(X2 *[FftLengthBy2Plus1]float32, gate, rate float32, mu *[FftLengthBy2Plus1]float32) {
	gv := archsimd.BroadcastFloat32x4(gate)
	rv := archsimd.BroadcastFloat32x4(rate)
	var zero f32x4
	for k := 0; k < FftLengthBy2; k += 4 {
		x := kernLd4(X2, k)
		kernSt4(mu, k, rv.Div(x).IfElse(x.Greater(gv), zero))
	}
	const k = FftLengthBy2
	if X2[k] > gate {
		mu[k] = rate / X2[k]
	} else {
		mu[k] = 0
	}
}

func kernReverbScalar(r, p []float32, s, d float32) {
	p = p[:len(r)]
	sv := archsimd.BroadcastFloat32x4(s)
	dv := archsimd.BroadcastFloat32x4(d)
	j := 0
	for ; j+4 <= len(r); j += 4 {
		q := archsimd.LoadFloat32x4(p[j:]).Mul(sv)
		archsimd.LoadFloat32x4(r[j:]).Add(q).Mul(dv).Store(r[j:])
	}
	for ; j < len(r); j++ {
		r[j] = (r[j] + float32(p[j]*s)) * d
	}
}
