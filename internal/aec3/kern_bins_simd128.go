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
	if !kernHas128 {
		kernMinGainGeneric(w, minEchoPower, minGain)
		return
	}
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
	if !kernHas128 {
		kernNoiseInitialUpdateGeneric(n, ini)
		return
	}
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
	if !kernHas128 {
		kernRefinedHErrorGeneric(mu, X2, hError)
		return
	}
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
	if !kernHas128 {
		kernRefinedLeakGeneric(E2Refined, E2Coarse, erl, disallowLeakageDiverged, leakageConverged, leakageDiverged, errorFloor, errorCeil, hError)
		return
	}
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
	if !kernHas128 {
		kernCoarseMuGeneric(X2, gate, rate, mu)
		return
	}
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
	if !kernHas128 {
		kernReverbScalarGeneric(r, p, s, d)
		return
	}
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
