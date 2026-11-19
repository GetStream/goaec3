//go:build goexperiment.simd && (arm64 || amd64) && !purego

package bands

import "simd/archsimd"

// archsimd ThreeBandFilterBank. The scalar C++ is elementwise in every
// output sample: bsFilterCore forms out[k] as ((((0 + x0*h0) + x1*h1) +
// x2*h2) + x3*h3) with xi = x[k - inShift - Stride*i], the banding adds each
// filter's modulated output to the band in a fixed filter order, and the
// synthesis modulation and upsampling do the same per sample. The kernels
// vectorize across output samples and keep that per-sample operation order,
// so they are bit-identical to threeBandFilterBankRef on every lane.
//
// Each filter input is kept as a history buffer: MemorySize samples of state
// followed by the SplitBandSize new samples, so bsFilterCore's three state /
// input cases become one load pattern. The analysis filters that share a
// downsampling index see the same input, so one history per index holds the
// state all of them carry in the reference.
//
// NaN payloads: when both operands of an add are NaN, the result carries one
// of the two payloads, picked by operand position (arm64 and x86 differ),
// and the compiler may commute vector adds. Every NaN the filter bank
// creates from non-NaN values is the architecture's default NaN, so the
// payloads only diverge when a NaN reaches it from outside. A frame whose
// input or state holds a NaN or an infinity therefore runs the reference,
// which reproduces the C++ operand order, on the same state.

// ThreeBandFilterBank splits a 48 kHz frame into three 16 kHz bands and
// merges them back. The prototype low-pass filter has half bandwidth, so a
// DCT shifts it to the center frequencies [1/12, 3/12, 5/12].
type ThreeBandFilterBank struct {
	// analysisHist[phase] is the filter input for downsampling index phase.
	analysisHist [bsSubSampling][bsHistSize]float32
	// synthesisHist[i] is modulated input i (see bsSynthesisInput).
	synthesisHist [bsNumSynthesisInputs][bsHistSize]float32

	// ref runs when the CPU lacks the archsimd features (amd64 without AVX).
	ref threeBandFilterBankRef
}

// Analysis splits in (FullBandSize samples) into NumBands downsampled
// frequency bands in out, each of SplitBandSize samples.
func (f *ThreeBandFilterBank) Analysis(in *[FullBandSize]float32, out *[NumBands][]float32) {
	if !bsHasSIMD || !bsAnalysisFinite(&f.analysisHist, in) {
		f.analysisRef(in, out)
		return
	}
	bsAnalysisKernel(&f.analysisHist, in, (*[SplitBandSize]float32)(out[0]),
		(*[SplitBandSize]float32)(out[1]), (*[SplitBandSize]float32)(out[2]))
}

// analysisRef runs the reference on the state held in analysisHist.
func (f *ThreeBandFilterBank) analysisRef(in *[FullBandSize]float32, out *[NumBands][]float32) {
	for _, e := range bsSchedule {
		copy(f.ref.stateAnalysis[e.filter][:], f.analysisHist[e.phase][:MemorySize])
	}
	f.ref.analysis(in, out)
	for _, e := range bsSchedule {
		copy(f.analysisHist[e.phase][:MemorySize], f.ref.stateAnalysis[e.filter][:])
	}
}

// Synthesis merges the NumBands downsampled bands in (SplitBandSize samples
// each) into out (FullBandSize samples).
func (f *ThreeBandFilterBank) Synthesis(in *[NumBands][]float32, out *[FullBandSize]float32) {
	in0 := (*[SplitBandSize]float32)(in[0])
	in1 := (*[SplitBandSize]float32)(in[1])
	in2 := (*[SplitBandSize]float32)(in[2])
	if !bsHasSIMD || !bsSynthesisFinite(&f.synthesisHist, in0, in1, in2) {
		f.synthesisRef(in, out)
		return
	}
	bsSynthesisKernel(&f.synthesisHist, in0, in1, in2, out)
}

// synthesisRef runs the reference on the state held in synthesisHist.
func (f *ThreeBandFilterBank) synthesisRef(in *[NumBands][]float32, out *[FullBandSize]float32) {
	for fi, i := range bsSynthesisInput {
		copy(f.ref.stateSynthesis[fi][:], f.synthesisHist[i][:MemorySize])
	}
	f.ref.synthesis(in, out)
	// Filters that share an input end with the same state.
	for fi, i := range bsSynthesisInput {
		copy(f.synthesisHist[i][:MemorySize], f.ref.stateSynthesis[fi][:])
	}
}

// bsFinite4 accumulates x - x, which is +0 for finite x and NaN otherwise.
func bsFinite4(acc archsimd.Float32x4, x *[4]float32) archsimd.Float32x4 {
	v := archsimd.LoadFloat32x4Array(x)
	return acc.Add(v.Sub(v))
}

func bsFiniteResult(acc archsimd.Float32x4) bool {
	var a [4]float32
	acc.StoreArray(&a)
	return a[0] == a[0] && a[1] == a[1] && a[2] == a[2] && a[3] == a[3]
}

// bsFiniteState checks a MemorySize state: four overlapping vectors.
func bsFiniteState(acc archsimd.Float32x4, h *[bsHistSize]float32) archsimd.Float32x4 {
	acc = bsFinite4(acc, (*[4]float32)(h[0:4]))
	acc = bsFinite4(acc, (*[4]float32)(h[4:8]))
	acc = bsFinite4(acc, (*[4]float32)(h[8:12]))
	return bsFinite4(acc, (*[4]float32)(h[MemorySize-4:MemorySize]))
}

// bsAnalysisFinite reports whether the analysis input and state are finite.
// The input is accumulated in eight independent sums: only the presence of
// a NaN in a sum matters, and one sum would chain 120 additions.
func bsAnalysisFinite(hist *[bsSubSampling][bsHistSize]float32, in *[FullBandSize]float32) bool {
	var acc archsimd.Float32x4
	for d := range hist {
		acc = bsFiniteState(acc, &hist[d])
	}
	var a0, a1, a2, a3, a4, a5, a6, a7 archsimd.Float32x4
	for k := 0; k < FullBandSize-FullBandSize%32; k += 32 {
		p := (*[32]float32)(in[k : k+32])
		a0 = bsFinite4(a0, (*[4]float32)(p[0:4]))
		a1 = bsFinite4(a1, (*[4]float32)(p[4:8]))
		a2 = bsFinite4(a2, (*[4]float32)(p[8:12]))
		a3 = bsFinite4(a3, (*[4]float32)(p[12:16]))
		a4 = bsFinite4(a4, (*[4]float32)(p[16:20]))
		a5 = bsFinite4(a5, (*[4]float32)(p[20:24]))
		a6 = bsFinite4(a6, (*[4]float32)(p[24:28]))
		a7 = bsFinite4(a7, (*[4]float32)(p[28:32]))
	}
	for k := FullBandSize - FullBandSize%32; k < FullBandSize; k += 4 {
		acc = bsFinite4(acc, (*[4]float32)(in[k:k+4]))
	}
	return bsFiniteResult(acc.Add(a0.Add(a1).Add(a2.Add(a3))).Add(a4.Add(a5).Add(a6.Add(a7))))
}

// bsSynthesisFinite reports whether the synthesis input and state are
// finite, with six independent sums over the bands.
func bsSynthesisFinite(hist *[bsNumSynthesisInputs][bsHistSize]float32, in0, in1, in2 *[SplitBandSize]float32) bool {
	var acc archsimd.Float32x4
	for fi := range hist {
		acc = bsFiniteState(acc, &hist[fi])
	}
	var a0, a1, a2, a3, a4, a5 archsimd.Float32x4
	for k := 0; k < SplitBandSize; k += 8 {
		p0 := (*[8]float32)(in0[k : k+8])
		p1 := (*[8]float32)(in1[k : k+8])
		p2 := (*[8]float32)(in2[k : k+8])
		a0 = bsFinite4(a0, (*[4]float32)(p0[0:4]))
		a1 = bsFinite4(a1, (*[4]float32)(p0[4:8]))
		a2 = bsFinite4(a2, (*[4]float32)(p1[0:4]))
		a3 = bsFinite4(a3, (*[4]float32)(p1[4:8]))
		a4 = bsFinite4(a4, (*[4]float32)(p2[0:4]))
		a5 = bsFinite4(a5, (*[4]float32)(p2[4:8]))
	}
	return bsFiniteResult(acc.Add(a0.Add(a1).Add(a2.Add(a3))).Add(a4.Add(a5)))
}

// bsCoeffs holds the broadcast coefficients of one filter.
type bsCoeffs struct{ c0, c1, c2, c3 archsimd.Float32x4 }

func bsBroadcastCoeffs(filter int) bsCoeffs {
	c := &bsFilterCoeffs[filter]
	return bsCoeffs{archsimd.BroadcastFloat32x4(c[0]), archsimd.BroadcastFloat32x4(c[1]),
		archsimd.BroadcastFloat32x4(c[2]), archsimd.BroadcastFloat32x4(c[3])}
}

// bsTap4 is bsFilterCore for out[k:k+4]: w is the history window
// h[k : k+19] and c the coefficients of a filter with shift inShift.
func bsTap4(w *[19]float32, inShift int, c *bsCoeffs) archsimd.Float32x4 {
	var acc archsimd.Float32x4
	acc = acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[15-inShift : 19-inShift])).Mul(c.c0))
	acc = acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[11-inShift : 15-inShift])).Mul(c.c1))
	acc = acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[7-inShift : 11-inShift])).Mul(c.c2))
	return acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[3-inShift : 7-inShift])).Mul(c.c3))
}

// bsAnalysisSIMD filters and bands sample-major: every filter of four output
// samples is evaluated, in the reference filter order, before the next four,
// so the band sums stay in registers. The modulation by bsDctModulation is
// written out per filter.
//
// For finite operands these rewrites are exact:
//   - m*v for m in {1, -1} is v and -v; (-m)*v is -(m*v); a + (-b) is a - b.
//   - Equal products are computed once.
//   - A term 0*v is +-0, and adding it leaves a sum unchanged unless the sum
//     is -0. The sums start as 0 + p (never -0) and a sum of two values is
//     -0 only when both are -0, so the term is dropped.
//
// With finite input and state every analysis filter output is finite (the
// taps of a filter sum to less than one in magnitude), and the synthesis
// modulation only rewrites operations on the finite band samples.
func bsAnalysisSIMD(hist *[bsSubSampling][bsHistSize]float32, in *[FullBandSize]float32,
	out0, out1, out2 *[SplitBandSize]float32) {
	// Downsample: phase d takes in[(bsSubSampling-1) - d + bsSubSampling*k].
	h0, h1, h2 := &hist[0], &hist[1], &hist[2]
	for k := 0; k < SplitBandSize; k++ {
		x := (*[3]float32)(in[3*k : 3*k+3])
		h0[MemorySize+k] = x[2]
		h1[MemorySize+k] = x[1]
		h2[MemorySize+k] = x[0]
	}

	var c [NumNonZeroFilters]bsCoeffs
	for i := range c {
		c[i] = bsBroadcastCoeffs(i)
	}
	two := archsimd.BroadcastFloat32x4(2)
	sqrt3 := archsimd.BroadcastFloat32x4(bsSqrt3)
	var zero archsimd.Float32x4
	for k := 0; k < SplitBandSize; k += 4 {
		w0 := (*[19]float32)(h0[k : k+19])
		w1 := (*[19]float32)(h1[k : k+19])
		w2 := (*[19]float32)(h2[k : k+19])

		// Phase 0: filters 0 {2,2,2} and 5 {-2,-2,-2}.
		t := two.Mul(bsTap4(w0, 0, &c[0]))
		o0 := zero.Add(t)
		o1 := o0
		o2 := o0
		t = two.Mul(bsTap4(w0, 2, &c[5]))
		o0 = o0.Sub(t)
		o1 = o1.Sub(t)
		o2 = o2.Sub(t)

		// Phase 1: filters 1 {s,0,-s}, 3 {-1,2,-1}, 6 {-s,0,s}, 8 {1,-2,1}.
		t = sqrt3.Mul(bsTap4(w1, 0, &c[1]))
		o0 = o0.Add(t)
		o2 = o2.Sub(t)
		v := bsTap4(w1, 1, &c[3])
		o0 = o0.Sub(v)
		o1 = o1.Add(two.Mul(v))
		o2 = o2.Sub(v)
		t = sqrt3.Mul(bsTap4(w1, 2, &c[6]))
		o0 = o0.Sub(t)
		o2 = o2.Add(t)
		v = bsTap4(w1, 3, &c[8])
		o0 = o0.Add(v)
		o1 = o1.Sub(two.Mul(v))
		o2 = o2.Add(v)

		// Phase 2: filters 2 {1,-2,1}, 4 {-s,0,s}, 7 {-1,2,-1}, 9 {s,0,-s}.
		v = bsTap4(w2, 0, &c[2])
		o0 = o0.Add(v)
		o1 = o1.Sub(two.Mul(v))
		o2 = o2.Add(v)
		t = sqrt3.Mul(bsTap4(w2, 1, &c[4]))
		o0 = o0.Sub(t)
		o2 = o2.Add(t)
		v = bsTap4(w2, 2, &c[7])
		o0 = o0.Sub(v)
		o1 = o1.Add(two.Mul(v))
		o2 = o2.Sub(v)
		t = sqrt3.Mul(bsTap4(w2, 3, &c[9]))
		o0 = o0.Add(t)
		o2 = o2.Sub(t)

		o0.StoreArray((*[4]float32)(out0[k : k+4]))
		o1.StoreArray((*[4]float32)(out1[k : k+4]))
		o2.StoreArray((*[4]float32)(out2[k : k+4]))
	}

	for d := range hist {
		copy(hist[d][:MemorySize], hist[d][SplitBandSize:])
	}
}

func bsSynthesisSIMD(hist *[bsNumSynthesisInputs][bsHistSize]float32,
	in0, in1, in2 *[SplitBandSize]float32, out *[FullBandSize]float32) {
	// Modulate: the filter input is ((0 + m0*b0) + m1*b1) + m2*b2.
	g0, g1, g2, g3, g4, g5 := &hist[0], &hist[1], &hist[2], &hist[3], &hist[4], &hist[5]
	two := archsimd.BroadcastFloat32x4(2)
	sqrt3 := archsimd.BroadcastFloat32x4(bsSqrt3)
	var zero archsimd.Float32x4
	for k := 0; k < SplitBandSize; k += 4 {
		b0 := archsimd.LoadFloat32x4Array((*[4]float32)(in0[k : k+4]))
		b1 := archsimd.LoadFloat32x4Array((*[4]float32)(in1[k : k+4]))
		b2 := archsimd.LoadFloat32x4Array((*[4]float32)(in2[k : k+4]))
		p0, p1, p2 := two.Mul(b0), two.Mul(b1), two.Mul(b2)
		q0, q2 := sqrt3.Mul(b0), sqrt3.Mul(b2)
		zero.Add(p0).Add(p1).Add(p2).StoreArray((*[4]float32)(g0[MemorySize+k : MemorySize+k+4]))
		zero.Sub(p0).Sub(p1).Sub(p2).StoreArray((*[4]float32)(g1[MemorySize+k : MemorySize+k+4]))
		zero.Add(q0).Sub(q2).StoreArray((*[4]float32)(g2[MemorySize+k : MemorySize+k+4]))
		zero.Sub(q0).Add(q2).StoreArray((*[4]float32)(g3[MemorySize+k : MemorySize+k+4]))
		zero.Add(b0).Sub(p1).Add(b2).StoreArray((*[4]float32)(g4[MemorySize+k : MemorySize+k+4]))
		zero.Sub(b0).Add(p1).Sub(b2).StoreArray((*[4]float32)(g5[MemorySize+k : MemorySize+k+4]))
	}

	// Filter, scale and upsample: phase d lands at out[d + bsSubSampling*k]
	// as the sum of upsamplingScaling * v over the phase's filters in inShift
	// order, starting from zero.
	var c [NumNonZeroFilters]bsCoeffs
	for i := range c {
		c[i] = bsBroadcastCoeffs(i)
	}
	scale := archsimd.BroadcastFloat32x4(bsSubSampling)
	for k := 0; k < SplitBandSize; k += 4 {
		w0 := (*[19]float32)(g0[k : k+19])
		w1 := (*[19]float32)(g1[k : k+19])
		w2 := (*[19]float32)(g2[k : k+19])
		w3 := (*[19]float32)(g3[k : k+19])
		w4 := (*[19]float32)(g4[k : k+19])
		w5 := (*[19]float32)(g5[k : k+19])

		// Phase 0: filters 0 and 5.
		o0 := zero.Add(scale.Mul(bsTap4(w0, 0, &c[0])))
		o0 = o0.Add(scale.Mul(bsTap4(w1, 2, &c[5])))

		// Phase 1: filters 1, 3, 6 and 8.
		o1 := zero.Add(scale.Mul(bsTap4(w2, 0, &c[1])))
		o1 = o1.Add(scale.Mul(bsTap4(w5, 1, &c[3])))
		o1 = o1.Add(scale.Mul(bsTap4(w3, 2, &c[6])))
		o1 = o1.Add(scale.Mul(bsTap4(w4, 3, &c[8])))

		// Phase 2: filters 2, 4, 7 and 9.
		o2 := zero.Add(scale.Mul(bsTap4(w4, 0, &c[2])))
		o2 = o2.Add(scale.Mul(bsTap4(w3, 1, &c[4])))
		o2 = o2.Add(scale.Mul(bsTap4(w5, 2, &c[7])))
		o2 = o2.Add(scale.Mul(bsTap4(w2, 3, &c[9])))

		var t0, t1, t2 [4]float32
		o0.StoreArray(&t0)
		o1.StoreArray(&t1)
		o2.StoreArray(&t2)
		y := (*[12]float32)(out[3*k : 3*k+12])
		y[0], y[1], y[2] = t0[0], t1[0], t2[0]
		y[3], y[4], y[5] = t0[1], t1[1], t2[1]
		y[6], y[7], y[8] = t0[2], t1[2], t2[2]
		y[9], y[10], y[11] = t0[3], t1[3], t2[3]
	}

	for i := range hist {
		copy(hist[i][:MemorySize], hist[i][SplitBandSize:])
	}
}
