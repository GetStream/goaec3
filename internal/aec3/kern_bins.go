package aec3

// Elementwise helpers for the per-block bin loops (65-bin spectra, 64-sample
// blocks). Each output element is one IEEE operation (or a fixed sequence of
// them) on the corresponding inputs, so the archsimd kernels in
// kern_bins_simd_{arm64,amd64}.go are bit-identical to these references in
// every lane. The references are the spec and the !goexperiment.simd /
// purego implementation.

// kernScaleGeneric sets z = x*a element-wise.
func kernScaleGeneric(x []float32, a float32, z []float32) {
	z = z[:len(x)]
	for i, v := range x {
		z[i] = v * a
	}
}

// kernAddGeneric sets z = x+y element-wise.
func kernAddGeneric(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	for i, v := range x {
		z[i] = v + y[i]
	}
}

// kernSubGeneric sets z = x-y element-wise.
func kernSubGeneric(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	for i, v := range x {
		z[i] = v - y[i]
	}
}

// kernSumRowsGeneric sets dst to the sum of the rows, accumulated per bin in
// row order starting from +0.
func kernSumRowsGeneric(dst *[FftLengthBy2Plus1]float32, rows []*[FftLengthBy2Plus1]float32) {
	*dst = [FftLengthBy2Plus1]float32{}
	for _, r := range rows {
		for k := range dst {
			dst[k] += r[k]
		}
	}
}

// kernUnpackGeneric is FftData::CopyFromPackedArray.
func kernUnpackGeneric(v *[FftLength]float32, re, im *[FftLengthBy2Plus1]float32) {
	re[0] = v[0]
	re[FftLengthBy2] = v[1]
	im[0] = 0
	im[FftLengthBy2] = 0
	for k, j := 1, 2; k < FftLengthBy2; k, j = k+1, j+2 {
		re[k] = v[j]
		im[k] = v[j+1]
	}
}

// kernPackGeneric is FftData::CopyToPackedArray.
func kernPackGeneric(re, im *[FftLengthBy2Plus1]float32, v *[FftLength]float32) {
	v[0] = re[0]
	v[1] = re[FftLengthBy2]
	for k, j := 1, 2; k < FftLengthBy2; k, j = k+1, j+2 {
		v[j] = re[k]
		v[j+1] = im[k]
	}
}

// kernMinMaxGeneric returns the minimum and maximum of x (len(x) > 0) as a
// loop from x[0] with "if v < mn" and "if v > mx" finds them.
func kernMinMaxGeneric(x []float32) (mn, mx float32) {
	mn, mx = x[0], x[0]
	for _, v := range x[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

// kernClampGeneric sets x = std::min(std::max(x, lo), hi) element-wise
// (SuppressionGain::GetMaxGain). For nonzero bounds lo < hi this is also
// SafeClamp(x, lo, hi) on every input (a value equal to a bound is that
// bound, and NaN passes through both), as the subtractor and the
// suppression filter use it.
func kernClampGeneric(x []float32, lo, hi float32) {
	for i, v := range x {
		x[i] = stdMin(stdMax(v, lo), hi)
	}
}

// The helpers below have archsimd kernels on arm64 and amd64 only; other builds run
// these references (kern_bins_other.go).

// kernMulAdd2Generic sets z = float32(a*b) + float32(c*d) element-wise.
func kernMulAdd2Generic(a, b, c, d, z []float32) {
	n := len(z)
	a, b, c, d = a[:n], b[:n], c[:n], d[:n]
	for i := range z {
		z[i] = float32(a[i]*b[i]) + float32(c[i]*d[i])
	}
}

// kernAxpyGeneric sets z += float32(x*a) element-wise.
func kernAxpyGeneric(x []float32, a float32, z []float32) {
	z = z[:len(x)]
	for i, v := range x {
		z[i] += float32(v * a)
	}
}

// kernOneMinusSqGeneric sets z = 1 - float32(x*x) element-wise.
func kernOneMinusSqGeneric(x, z []float32) {
	z = z[:len(x)]
	for i, v := range x {
		z[i] = 1 - float32(v*v)
	}
}

// kernDiffPowerGeneric sets z = float32(dr*dr) + float32(di*di) with
// dr = yr-er and di = yi-ei.
func kernDiffPowerGeneric(yr, er, yi, ei, z []float32) {
	n := len(z)
	yr, er, yi, ei = yr[:n], er[:n], yi[:n], ei[:n]
	for k := range z {
		dr := yr[k] - er[k]
		di := yi[k] - ei[k]
		z[k] = float32(dr*dr) + float32(di*di)
	}
}

// kernStdMinGeneric sets z = std::min(x, y) = (y < x ? y : x) element-wise.
func kernStdMinGeneric(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	for i, v := range x {
		z[i] = stdMin(v, y[i])
	}
}

// kernStdMaxScalarGeneric sets x = std::max(x, a) = (x < a ? a : x)
// element-wise.
func kernStdMaxScalarGeneric(x []float32, a float32) {
	for i, v := range x {
		x[i] = stdMax(v, a)
	}
}

// kernSmoothGeneric sets s = s + float32(c*(y-s)) element-wise.
func kernSmoothGeneric(s, y []float32, c float32) {
	y = y[:len(s)]
	for i, a := range s {
		s[i] = a + float32(c*(y[i]-a))
	}
}

// kernNoAudibleEchoGeneric is the per-bin loop of
// SuppressionGain::GainToNoAudibleEcho.
func kernNoAudibleEchoGeneric(nearend, echo, masker, enrT, enrS, emrT, gain *[FftLengthBy2Plus1]float32) {
	for k := range gain {
		enr := echo[k] / (nearend[k] + 1) // Echo-to-nearend ratio.
		emr := echo[k] / (masker[k] + 1)  // Echo-to-masker (noise) ratio.
		g := float32(1)
		if enr > enrT[k] && emr > emrT[k] {
			g = (enrS[k] - enr) / (enrS[k] - enrT[k])
			g = stdMax(g, emrT[k]/emr)
		}
		gain[k] = g
	}
}

// kernMinGainGeneric sets minGain = std::min(w > 0 ? minEchoPower/w : 1, 1).
func kernMinGainGeneric(w *[FftLengthBy2Plus1]float32, minEchoPower float32, minGain *[FftLengthBy2Plus1]float32) {
	for k := range minGain {
		if w[k] > 0 {
			minGain[k] = minEchoPower / w[k]
		} else {
			minGain[k] = 1
		}
		minGain[k] = stdMin(minGain[k], 1)
	}
}

// kernWeighGeneric is the loop of sgWeigh over echo/weightedEcho:
// echo * std::max(0.f, 1.f - tmp * tmp), where a NaN tmp gives a factor of 0.
func kernWeighGeneric(threshold, normalizer float32, echo, weightedEcho []float32) {
	weightedEcho = weightedEcho[:len(echo)]
	for k, e := range echo {
		if e < threshold {
			tmp := float32((threshold - e) * normalizer)
			weightedEcho[k] = float32(e * stdMax(0, 1-float32(tmp*tmp)))
		} else {
			weightedEcho[k] = e
		}
	}
}

// kernNoiseUpdateGeneric is the N2 update of ComfortNoiseGenerator::Compute.
func kernNoiseUpdateGeneric(n, s *[FftLengthBy2Plus1]float32) {
	for k := range n {
		a, b := n[k], s[k]
		if b < a {
			n[k] = (float32(0.9*b) + float32(0.1*a)) * 1.0002
		} else {
			n[k] = a * 1.0002
		}
	}
}

// kernNoiseInitialUpdateGeneric is the N2_initial update of
// ComfortNoiseGenerator::Compute.
func kernNoiseInitialUpdateGeneric(n, ini *[FftLengthBy2Plus1]float32) {
	for k := range n {
		a, b := n[k], ini[k]
		if a > b {
			ini[k] = b + float32(0.001*(a-b))
		} else {
			ini[k] = a
		}
	}
}

// kernFir3Generic sets out[k] = ((+0 + x[k]*h0) + x[k-1]*h1) + x[k-2]*h2 for
// k in [from, to] (from >= 2).
func kernFir3Generic(x []float32, h *[3]float32, out []float32, from, to int) {
	for k := from; k <= to; k++ {
		var tmp float32
		for j := range h {
			tmp += float32(x[k-j] * h[j])
		}
		out[k] = tmp
	}
}

// kernDivGeneric sets z = x/y element-wise.
func kernDivGeneric(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	for i, v := range x {
		z[i] = v / y[i]
	}
}

// kernReverbGeneric sets r = (r + float32(p*s)) * d element-wise.
func kernReverbGeneric(r, p, s []float32, d float32) {
	p = p[:len(r)]
	s = s[:len(r)]
	for k := range r {
		r[k] = (r[k] + float32(p[k]*s[k])) * d
	}
}

// kernStdMaxGeneric sets z = std::max(x, y) = (x < y ? y : x) element-wise.
func kernStdMaxGeneric(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	for i, v := range x {
		z[i] = stdMax(v, y[i])
	}
}

// kernRefinedMuGeneric is the mu loop of RefinedFilterUpdateGain::Compute:
// mu = X2 >= gate ? h / (float32(float32(0.5*h)*X2) + float32(sp*E2)) : +0.
func kernRefinedMuGeneric(X2, hError, E2 *[FftLengthBy2Plus1]float32, gate, sp float32, mu *[FftLengthBy2Plus1]float32) {
	for k := range mu {
		if X2[k] >= gate {
			h := hError[k]
			mu[k] = h / (float32(float32(0.5*h)*X2[k]) + float32(sp*E2[k]))
		} else {
			mu[k] = 0
		}
	}
}

// kernRefinedHErrorGeneric sets h -= float32(float32(float32(0.5*mu)*X2)*h).
func kernRefinedHErrorGeneric(mu, X2, hError *[FftLengthBy2Plus1]float32) {
	for k := range hError {
		hError[k] -= float32(float32(float32(0.5*mu[k])*X2[k]) * hError[k])
	}
}

// kernRefinedLeakGeneric is the H_error leakage update of
// RefinedFilterUpdateGain::Compute.
func kernRefinedLeakGeneric(E2Refined, E2Coarse *[FftLengthBy2Plus1]float32, erl []float32, disallowLeakageDiverged bool,
	leakageConverged, leakageDiverged, errorFloor, errorCeil float32, hError *[FftLengthBy2Plus1]float32) {
	erl = erl[:FftLengthBy2Plus1]
	for k := range hError {
		if E2Refined[k] <= E2Coarse[k] || disallowLeakageDiverged {
			hError[k] += float32(leakageConverged * erl[k])
		} else {
			hError[k] += float32(leakageDiverged * erl[k])
		}
		hError[k] = stdMax(hError[k], errorFloor)
		hError[k] = stdMin(hError[k], errorCeil)
	}
}

// kernCoarseMuGeneric sets mu = X2 > gate ? rate/X2 : +0.
func kernCoarseMuGeneric(X2 *[FftLengthBy2Plus1]float32, gate, rate float32, mu *[FftLengthBy2Plus1]float32) {
	for k := range mu {
		if X2[k] > gate {
			mu[k] = rate / X2[k]
		} else {
			mu[k] = 0
		}
	}
}

// kernReverbScalarGeneric sets r = (r + float32(p*s)) * d element-wise.
func kernReverbScalarGeneric(r, p []float32, s, d float32) {
	p = p[:len(r)]
	for k := range r {
		r[k] = (r[k] + float32(p[k]*s)) * d
	}
}
