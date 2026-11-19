//go:build goexperiment.simd && (arm64 || amd64) && !purego

package aec3

import "simd/archsimd"

// 128-bit archsimd kernels for the kern_bins.go helpers, shared by arm64
// (NEON) and amd64 (VEX-encoded SSE operations, which need AVX; without it
// they run the references). Every lane is one IEEE operation, or a compare
// and a select, per element, as in the reference.

type f32x4 = archsimd.Float32x4

// vStdMin is std::min(a, b) per lane: b < a ? b : a. FMIN differs from it
// on NaN and on equal zeros, so it is a compare and a select.
func vStdMin(a, b f32x4) f32x4 { return b.IfElse(b.Less(a), a) }

// vStdMax is std::max(a, b) per lane: a < b ? b : a.
func vStdMax(a, b f32x4) f32x4 { return b.IfElse(a.Less(b), a) }

func kernMulAdd2(a, b, c, d, z []float32) {
	if !kernHas128 {
		kernMulAdd2Generic(a, b, c, d, z)
		return
	}
	n := len(z)
	a, b, c, d = a[:n], b[:n], c[:n], d[:n]
	if kernIs64(n) {
		if n == 65 {
			z[64] = float32(a[64]*b[64]) + float32(c[64]*d[64])
		}
		kernMulAdd2x64((*[64]float32)(a), (*[64]float32)(b), (*[64]float32)(c), (*[64]float32)(d), (*[64]float32)(z))
		return
	}
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
	if !kernHas128 {
		kernAxpyGeneric(x, a, z)
		return
	}
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] += float32(x[64] * a)
		}
		kernAxpy64((*[64]float32)(x), a, (*[64]float32)(z))
		return
	}
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
	if !kernHas128 {
		kernOneMinusSqGeneric(x, z)
		return
	}
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = 1 - float32(x[64]*x[64])
		}
		kernOneMinusSq64((*[64]float32)(x), (*[64]float32)(z))
		return
	}
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
	if !kernHas128 {
		kernDiffPowerGeneric(yr, er, yi, ei, z)
		return
	}
	n := len(z)
	yr, er, yi, ei = yr[:n], er[:n], yi[:n], ei[:n]
	if kernIs64(n) {
		if n == 65 {
			dr := yr[64] - er[64]
			di := yi[64] - ei[64]
			z[64] = float32(dr*dr) + float32(di*di)
		}
		kernDiffPower64((*[64]float32)(yr), (*[64]float32)(er), (*[64]float32)(yi), (*[64]float32)(ei), (*[64]float32)(z))
		return
	}
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
	if !kernHas128 {
		kernStdMinGeneric(x, y, z)
		return
	}
	y = y[:len(x)]
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = stdMin(x[64], y[64])
		}
		kernStdMin64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		return
	}
	j := 0
	for ; j+4 <= len(x); j += 4 {
		vStdMin(archsimd.LoadFloat32x4(x[j:]), archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = stdMin(x[j], y[j])
	}
}

func kernStdMaxScalar(x []float32, a float32) {
	if !kernHas128 {
		kernStdMaxScalarGeneric(x, a)
		return
	}
	if kernIs64(len(x)) {
		if len(x) == 65 {
			x[64] = stdMax(x[64], a)
		}
		kernStdMaxScalar64((*[64]float32)(x), a)
		return
	}
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
	if !kernHas128 {
		kernSmoothGeneric(s, y, c)
		return
	}
	y = y[:len(s)]
	if kernIs64(len(s)) {
		if len(s) == 65 {
			s[64] += float32(c * (y[64] - s[64]))
		}
		kernSmooth64((*[64]float32)(s), (*[64]float32)(y), c)
		return
	}
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
	if !kernHas128 {
		kernNoAudibleEchoGeneric(nearend, echo, masker, enrT, enrS, emrT, gain)
		return
	}
	one := archsimd.BroadcastFloat32x4(1)
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		e := archsimd.LoadFloat32x4Array((*[4]float32)(echo[k : k+4]))
		enr := e.Div(archsimd.LoadFloat32x4Array((*[4]float32)(nearend[k : k+4])).Add(one))
		emr := e.Div(archsimd.LoadFloat32x4Array((*[4]float32)(masker[k : k+4])).Add(one))
		et := archsimd.LoadFloat32x4Array((*[4]float32)(enrT[k : k+4]))
		es := archsimd.LoadFloat32x4Array((*[4]float32)(enrS[k : k+4]))
		mt := archsimd.LoadFloat32x4Array((*[4]float32)(emrT[k : k+4]))
		g := vStdMax(es.Sub(enr).Div(es.Sub(et)), mt.Div(emr))
		g.IfElse(enr.Greater(et).And(emr.Greater(mt)), one).StoreArray((*[4]float32)(gain[k : k+4]))
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
	if !kernHas128 {
		kernMinGainGeneric(w, minEchoPower, minGain)
		return
	}
	one := archsimd.BroadcastFloat32x4(1)
	var zero f32x4
	mp := archsimd.BroadcastFloat32x4(minEchoPower)
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		wv := archsimd.LoadFloat32x4Array((*[4]float32)(w[k : k+4]))
		vStdMin(mp.Div(wv).IfElse(wv.Greater(zero), one), one).StoreArray((*[4]float32)(minGain[k : k+4]))
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
	const n = FftLengthBy2
	if w[n] > 0 {
		minGain[n] = minEchoPower / w[n]
	} else {
		minGain[n] = 1
	}
	minGain[n] = stdMin(minGain[n], 1)
}

func kernWeigh(threshold, normalizer float32, echo, weightedEcho []float32) {
	if !kernHas128 {
		kernWeighGeneric(threshold, normalizer, echo, weightedEcho)
		return
	}
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
	if !kernHas128 {
		kernNoiseUpdateGeneric(n, s)
		return
	}
	c09 := archsimd.BroadcastFloat32x4(0.9)
	c01 := archsimd.BroadcastFloat32x4(0.1)
	c1 := archsimd.BroadcastFloat32x4(1.0002)
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		p := (*[4]float32)(n[k : k+4])
		a := archsimd.LoadFloat32x4Array(p)
		b := archsimd.LoadFloat32x4Array((*[4]float32)(s[k : k+4]))
		u := c09.Mul(b).Add(c01.Mul(a)).Mul(c1)
		u.IfElse(b.Less(a), a.Mul(c1)).StoreArray(p)
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
	if !kernHas128 {
		kernNoiseInitialUpdateGeneric(n, ini)
		return
	}
	c := archsimd.BroadcastFloat32x4(0.001)
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		p := (*[4]float32)(ini[k : k+4])
		a := archsimd.LoadFloat32x4Array((*[4]float32)(n[k : k+4]))
		b := archsimd.LoadFloat32x4Array(p)
		b.Add(c.Mul(a.Sub(b))).IfElse(a.Greater(b), a).StoreArray(p)
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
	const k = FftLengthBy2
	a, b := n[k], ini[k]
	if a > b {
		ini[k] = b + float32(0.001*(a-b))
	} else {
		ini[k] = a
	}
}

func kernFir3(x []float32, h *[3]float32, out []float32, from, to int) {
	if !kernHas128 {
		kernFir3Generic(x, h, out, from, to)
		return
	}
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
	if !kernHas128 {
		kernDivGeneric(x, y, z)
		return
	}
	y = y[:len(x)]
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = x[64] / y[64]
		}
		kernDiv64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		return
	}
	j := 0
	for ; j+4 <= len(x); j += 4 {
		archsimd.LoadFloat32x4(x[j:]).Div(archsimd.LoadFloat32x4(y[j:])).Store(z[j:])
	}
	for ; j < len(x); j++ {
		z[j] = x[j] / y[j]
	}
}

func kernReverb(r, p, s []float32, d float32) {
	if !kernHas128 {
		kernReverbGeneric(r, p, s, d)
		return
	}
	p = p[:len(r)]
	s = s[:len(r)]
	if kernIs64(len(r)) {
		if len(r) == 65 {
			r[64] = (r[64] + float32(p[64]*s[64])) * d
		}
		kernReverb64((*[64]float32)(r), (*[64]float32)(p), (*[64]float32)(s), d)
		return
	}
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
	if !kernHas128 {
		kernStdMaxGeneric(x, y, z)
		return
	}
	y = y[:len(x)]
	z = z[:len(x)]
	if kernIs64(len(x)) {
		if len(x) == 65 {
			z[64] = stdMax(x[64], y[64])
		}
		kernStdMax64((*[64]float32)(x), (*[64]float32)(y), (*[64]float32)(z))
		return
	}
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
	if !kernHas128 {
		kernRefinedMuGeneric(X2, hError, E2, gate, sp, mu)
		return
	}
	half := archsimd.BroadcastFloat32x4(0.5)
	gv := archsimd.BroadcastFloat32x4(gate)
	spv := archsimd.BroadcastFloat32x4(sp)
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		x := kernLd4(X2, k)
		h := kernLd4(hError, k)
		d := half.Mul(h).Mul(x).Add(spv.Mul(kernLd4(E2, k)))
		kernSt4(mu, k, h.Div(d).IfElse(x.GreaterEqual(gv), zero))
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
	const k = FftLengthBy2
	if X2[k] >= gate {
		h := hError[k]
		mu[k] = h / (float32(float32(0.5*h)*X2[k]) + float32(sp*E2[k]))
	} else {
		mu[k] = 0
	}
}

func kernRefinedHError(mu, X2, hError *[FftLengthBy2Plus1]float32) {
	if !kernHas128 {
		kernRefinedHErrorGeneric(mu, X2, hError)
		return
	}
	half := archsimd.BroadcastFloat32x4(0.5)
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		h := kernLd4(hError, k)
		kernSt4(hError, k, h.Sub(half.Mul(kernLd4(mu, k)).Mul(kernLd4(X2, k)).Mul(h)))
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
	const k = FftLengthBy2
	hError[k] -= float32(float32(float32(0.5*mu[k])*X2[k]) * hError[k])
}

// kernRefinedLeak selects the leakage per lane.
func kernRefinedLeak(E2Refined, E2Coarse *[FftLengthBy2Plus1]float32, erl []float32, disallowLeakageDiverged bool,
	leakageConverged, leakageDiverged, errorFloor, errorCeil float32, hError *[FftLengthBy2Plus1]float32) {
	if !kernHas128 {
		kernRefinedLeakGeneric(E2Refined, E2Coarse, erl, disallowLeakageDiverged, leakageConverged, leakageDiverged, errorFloor, errorCeil, hError)
		return
	}
	e := (*[FftLengthBy2Plus1]float32)(erl)
	lc := archsimd.BroadcastFloat32x4(leakageConverged)
	ld := archsimd.BroadcastFloat32x4(leakageDiverged)
	fl := archsimd.BroadcastFloat32x4(errorFloor)
	ce := archsimd.BroadcastFloat32x4(errorCeil)
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		l := lc
		if !disallowLeakageDiverged {
			l = lc.IfElse(kernLd4(E2Refined, k).LessEqual(kernLd4(E2Coarse, k)), ld)
		}
		h := kernLd4(hError, k).Add(l.Mul(kernLd4(e, k)))
		kernSt4(hError, k, vStdMin(vStdMax(h, fl), ce))
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
	if !kernHas128 {
		kernCoarseMuGeneric(X2, gate, rate, mu)
		return
	}
	gv := archsimd.BroadcastFloat32x4(gate)
	rv := archsimd.BroadcastFloat32x4(rate)
	var zero f32x4
	f := func(k int) {
		_ = zero // a captured vector makes the compiler inline the closure
		x := kernLd4(X2, k)
		kernSt4(mu, k, rv.Div(x).IfElse(x.Greater(gv), zero))
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
	const k = FftLengthBy2
	if X2[k] > gate {
		mu[k] = rate / X2[k]
	} else {
		mu[k] = 0
	}
}

func kernReverbScalar(r, p []float32, s, d float32) {
	if !kernHas128 {
		kernReverbScalarGeneric(r, p, s, d)
		return
	}
	p = p[:len(r)]
	if kernIs64(len(r)) {
		if len(r) == 65 {
			r[64] = (r[64] + float32(p[64]*s)) * d
		}
		kernReverbScalar64((*[64]float32)(r), (*[64]float32)(p), s, d)
		return
	}
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

// kernErlUpdate evaluates the update and the release of every bin and
// selects per lane. Bin 0 shares the first vector: its ERL is replaced by
// the caller (erl[0] = erl[1]) and its hold counter is put back to zero.
func kernErlUpdate(X2, Y2, erl *[FftLengthBy2Plus1]float32, hold *[FftLengthBy2]int32) {
	if !kernHas128 {
		kernErlUpdateGeneric(X2, Y2, erl, hold)
		return
	}
	x2Min := archsimd.BroadcastFloat32x4(stErlX2Min)
	c01 := archsimd.BroadcastFloat32x4(0.1)
	two := archsimd.BroadcastFloat32x4(2)
	minErl := archsimd.BroadcastFloat32x4(stMinErl)
	maxErl := archsimd.BroadcastFloat32x4(stMaxErl)
	hold1000 := archsimd.BroadcastInt32x4(1000)
	one := archsimd.BroadcastInt32x4(1)
	var zero archsimd.Int32x4
	f := func(k int) {
		_ = x2Min // a captured vector makes the compiler inline the closure
		x := kernLd4(X2, k)
		e := kernLd4(erl, k)
		hp := (*[4]int32)(hold[k : k+4])
		h := archsimd.LoadInt32x4Array(hp)
		newErl := kernLd4(Y2, k).Div(x)
		update := x.Greater(x2Min).And(newErl.Less(e))
		e = vStdMax(e.Add(c01.Mul(newErl.Sub(e))), minErl).IfElse(update, e)
		h = hold1000.IfElse(update, h).Sub(one)
		h.StoreArray(hp)
		kernSt4(erl, k, vStdMin(maxErl, two.Mul(e)).IfElse(h.LessEqual(zero), e))
	}
	g := func(k int) {
		_ = x2Min
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
	hold[0] = 0
}

// kernPeakBin keeps the first maximum of each lane and its bin, then picks
// the lane with the largest maximum and, among equals, the lowest bin. A
// later NaN never exceeds a lane maximum, as in the scalar search; a NaN in
// the first vector makes a lane maximum NaN, and that case runs the scalar
// search.
func kernPeakBin(x *[FftLengthBy2Plus1]float32) int {
	if !kernHas128 {
		return kernPeakBinGeneric(x)
	}
	best := kernLd4(x, 0)
	bins := archsimd.LoadInt32x4Array(&[4]int32{0, 1, 2, 3})
	bestBins := bins
	four := archsimd.BroadcastInt32x4(4)
	f := func(k int) {
		_ = four // a captured vector makes the compiler inline the closure
		v := kernLd4(x, k)
		bins = bins.Add(four)
		gt := v.Greater(best)
		best = v.IfElse(gt, best)
		bestBins = bins.IfElse(gt, bestBins)
	}
	f(4)
	f(8)
	f(12)
	f(16)
	f(20)
	f(24)
	f(28)
	f(32)
	f(36)
	f(40)
	f(44)
	f(48)
	f(52)
	f(56)
	f(60)
	var v [4]float32
	var b [4]int32
	best.StoreArray(&v)
	bestBins.StoreArray(&b)
	if v[0] != v[0] || v[1] != v[1] || v[2] != v[2] || v[3] != v[3] {
		return kernPeakBinGeneric(x)
	}
	p := 0
	for l := 1; l < 4; l++ {
		if v[l] > v[p] || v[l] == v[p] && b[l] < b[p] {
			p = l
		}
	}
	if x[FftLengthBy2] > v[p] {
		return FftLengthBy2
	}
	return int(b[p])
}

// kernNarrowBands compares four bins at a time; the last step covers bins
// 60 to 63 and overlaps the one before, which the OR allows.
func kernNarrowBands(x *[FftLengthBy2Plus1]float32, narrow *[FftLengthBy2]int32) {
	if !kernHas128 {
		kernNarrowBandsGeneric(x, narrow)
		return
	}
	three := archsimd.BroadcastFloat32x4(3)
	f := func(k int) {
		_ = three // a captured vector makes the compiler inline the closure
		// Bins k+1 to k+4: x[k:k+4] and x[k+2:k+6] are their neighbours.
		lo := archsimd.LoadFloat32x4Array((*[4]float32)(x[k : k+4]))
		mid := archsimd.LoadFloat32x4Array((*[4]float32)(x[k+1 : k+5]))
		hi := archsimd.LoadFloat32x4Array((*[4]float32)(x[k+2 : k+6]))
		np := (*[4]int32)(narrow[k : k+4])
		m := mid.Greater(three.Mul(vStdMax(lo, hi))).ToInt32x4()
		archsimd.LoadInt32x4Array(np).Or(m).StoreArray(np)
	}
	g := func(k int) {
		_ = three
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	f(48)
	f(52)
	f(56)
	f(59)
}

// kernNoiseFloorUpdate evaluates the three outcomes of a bin and selects
// per lane.
func kernNoiseFloorUpdate(renderPower, noiseFloor *[FftLengthBy2Plus1]float32, counter *[FftLengthBy2Plus1]int32,
	hold int32, minPower float32) {
	if !kernHas128 {
		kernNoiseFloorUpdateGeneric(renderPower, noiseFloor, counter, hold, minPower)
		return
	}
	holdV := archsimd.BroadcastInt32x4(hold)
	one := archsimd.BroadcastInt32x4(1)
	var zero archsimd.Int32x4
	c11 := archsimd.BroadcastFloat32x4(1.1)
	minV := archsimd.BroadcastFloat32x4(minPower)
	f := func(k int) {
		_ = c11 // a captured vector makes the compiler inline the closure
		p := kernLd4(renderPower, k)
		n := kernLd4(noiseFloor, k)
		cp := (*[4]int32)(counter[k : k+4])
		c := archsimd.LoadInt32x4Array(cp)
		lower := p.Less(n)
		held := holdV.LessEqual(c)
		kernSt4(noiseFloor, k, p.IfElse(lower, vStdMax(n.Mul(c11), minV).IfElse(held, n)))
		zero.IfElse(lower, c.IfElse(held, c.Add(one))).StoreArray(cp)
	}
	g := func(k int) {
		_ = c11
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
	const k = FftLengthBy2
	if renderPower[k] < noiseFloor[k] {
		noiseFloor[k] = renderPower[k]
		counter[k] = 0
	} else if counter[k] >= hold {
		noiseFloor[k] = stdMax(noiseFloor[k]*1.1, minPower)
	} else {
		counter[k]++
	}
}

func kernStdMinScalar(x []float32, a float32) {
	if !kernHas128 {
		kernStdMinScalarGeneric(x, a)
		return
	}
	av := archsimd.BroadcastFloat32x4(a)
	j := 0
	for ; j+4 <= len(x); j += 4 {
		vStdMin(archsimd.LoadFloat32x4(x[j:]), av).Store(x[j:])
	}
	for ; j < len(x); j++ {
		x[j] = stdMin(x[j], a)
	}
}

// kernSumMaxSq keeps the sum as the serial chain of additions it is and
// takes the maximum per lane. std::max(mx, x2) = (mx < x2 ? x2 : mx) keeps
// mx on a NaN x2 and stays NaN on a NaN mx, and every other x2 is +0 or
// positive, so the fold is the largest of mx and the non-NaN squares (or
// NaN) in any order: the four lanes start from mx and fold together at the
// end.
func kernSumMaxSq(x *[BlockSize]float32, sum, mx float32) (float32, float32) {
	if !kernHas128 {
		return kernSumMaxSqGeneric(x, sum, mx)
	}
	for _, v := range x {
		sum += float32(v * v)
	}
	m := archsimd.BroadcastFloat32x4(mx)
	f := func(k int) {
		_ = m // a captured vector makes the compiler inline the closure
		v := kernLd64(x, k)
		m = vStdMax(m, v.Mul(v))
	}
	g := func(k int) {
		_ = m
		f(k)
		f(k + 4)
		f(k + 8)
		f(k + 12)
	}
	g(0)
	g(16)
	g(32)
	g(48)
	var lanes [4]float32
	m.StoreArray(&lanes)
	return sum, stdMax(stdMax(lanes[0], lanes[1]), stdMax(lanes[2], lanes[3]))
}
