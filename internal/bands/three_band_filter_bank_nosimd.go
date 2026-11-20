//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package bands

// Pure-Go ThreeBandFilterBank. The scalar C++ is elementwise in every output
// sample: bsFilterCore forms out[k] as ((((0 + x0*h0) + x1*h1) + x2*h2) +
// x3*h3) with xi = x[k - inShift - Stride*i], the banding adds each filter's
// modulated output to the band in a fixed filter order, and the synthesis
// modulation and upsampling do the same per sample. The loops below run
// sample-major: every filter of one output sample is evaluated before the
// next sample, in the reference filter order, so the band and output sums
// stay in registers. Each value is computed by the same sequence of rounded
// operations as in threeBandFilterBankRef.
//
// Each filter input is kept as a history buffer: MemorySize samples of state
// followed by the SplitBandSize new samples, so bsFilterCore's three state /
// input cases become one tap pattern.
//
// The modulation by bsDctModulation is written out per filter. For finite
// operands these rewrites are exact:
//   - m*v for m in {1, -1} is v and -v; (-m)*v is -(m*v); a + (-b) is a - b.
//   - Equal products are computed once.
//   - A term 0*v is +-0, and adding it leaves a sum unchanged unless the sum
//     is -0. The sums start as 0 + p (never -0) and a sum of two values is
//     -0 only when both are -0, so the term is dropped.
//
// A filter output starts as 0 + x0*h0 in the C++. That addition changes
// the output only when it is -0 (all four products -0), into +0. An output
// reaches the result as a term t (itself, or its product with a positive
// constant, of the same sign) that is added to or subtracted from a sum
// which starts as 0 + t and so is never -0: a zero of either sign leaves
// such a sum as it is, and 0 + t is +0 for both. So the output is computed
// without the addition.
//
// The loops take one output sample per step on amd64, where a coefficient
// is the memory operand of its multiplication. Elsewhere a step is two
// samples: the coefficients of a filter are loaded once for both, and the
// history values, which the two samples read from adjacent elements, are
// loaded in pairs.
//
// With finite input and state every filter output is finite (the taps of a
// filter sum to less than one in magnitude) and no NaN is created, so the
// rewrites hold. A frame whose input or state holds a NaN or an infinity
// runs the reference on the same state, which also reproduces the C++
// operand order for NaN payloads.

// ThreeBandFilterBank splits a 48 kHz frame into three 16 kHz bands and
// merges them back. The prototype low-pass filter has half bandwidth, so a
// DCT shifts it to the center frequencies [1/12, 3/12, 5/12].
type ThreeBandFilterBank struct {
	// analysisHist[phase] is the filter input for downsampling index phase.
	analysisHist [bsSubSampling][bsHistSize]float32
	// synthesisHist[i] is modulated input i (see bsSynthesisInput).
	synthesisHist [bsNumSynthesisInputs][bsHistSize]float32

	// ref runs the frames that hold a NaN or an infinity.
	ref threeBandFilterBankRef
}

// Analysis splits in (FullBandSize samples) into NumBands downsampled
// frequency bands in out, each of SplitBandSize samples.
func (f *ThreeBandFilterBank) Analysis(in *[FullBandSize]float32, out *[NumBands][]float32) {
	// Downsample: phase d takes in[(bsSubSampling-1) - d + bsSubSampling*k].
	// x - x is +0 for finite x and NaN otherwise.
	h0, h1, h2 := &f.analysisHist[0], &f.analysisHist[1], &f.analysisHist[2]
	var n0, n1, n2 float32
	for k := 0; k < MemorySize; k++ {
		n0 += h0[k] - h0[k]
		n1 += h1[k] - h1[k]
		n2 += h2[k] - h2[k]
	}
	for k := 0; k < SplitBandSize; k++ {
		x := (*[3]float32)(in[3*k : 3*k+3])
		x0, x1, x2 := x[0], x[1], x[2]
		h0[MemorySize+k] = x2
		h1[MemorySize+k] = x1
		h2[MemorySize+k] = x0
		n0 += x0 - x0
		n1 += x1 - x1
		n2 += x2 - x2
	}
	if n := n0 + n1 + n2; n != n {
		f.analysisRef(in, out)
		return
	}
	bsAnalysisGo(&f.analysisHist, (*[SplitBandSize]float32)(out[0]),
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
	var n0, n1, n2 float32
	for i := range f.synthesisHist {
		h := &f.synthesisHist[i]
		for k := 0; k < MemorySize; k++ {
			n0 += h[k] - h[k]
		}
	}
	for k := range in0 {
		n0 += in0[k] - in0[k]
		n1 += in1[k] - in1[k]
		n2 += in2[k] - in2[k]
	}
	if n := n0 + n1 + n2; n != n {
		f.synthesisRef(in, out)
		return
	}
	bsSynthesisGo(&f.synthesisHist, in0, in1, in2, out)
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

// bsTap is bsFilterCore for one output sample, without its leading 0 +: w
// is the history window that ends at the newest sample the unshifted filter
// reads, c the coefficients.
func bsTap(w *[16]float32, inShift int, c *[FilterSize]float32) float32 {
	return float32(w[15-inShift]*c[0]) + float32(w[11-inShift]*c[1]) +
		float32(w[7-inShift]*c[2]) + float32(w[3-inShift]*c[3])
}

// bsTapAt is bsTap on a window with one more sample, for the two output
// samples of a step: the output at the window's first sample reads up to
// w[15-inShift] and the next one up to w[16-inShift], which is top. The two
// calls of a step inline, and the compiler loads the coefficients once.
func bsTapAt(w *[17]float32, top int, c *[FilterSize]float32) float32 {
	return float32(w[top]*c[0]) + float32(w[top-4]*c[1]) +
		float32(w[top-8]*c[2]) + float32(w[top-12]*c[3])
}

func bsAnalysisGo(hist *[bsSubSampling][bsHistSize]float32, out0, out1, out2 *[SplitBandSize]float32) {
	if bsIsAMD64 {
		bsAnalysis1Go(hist, out0, out1, out2)
	} else {
		bsAnalysis2Go(hist, out0, out1, out2)
	}
	for d := range hist {
		copy(hist[d][:MemorySize], hist[d][SplitBandSize:])
	}
}

// bsAnalysis1Go filters and bands one output sample per step.
func bsAnalysis1Go(hist *[bsSubSampling][bsHistSize]float32, out0, out1, out2 *[SplitBandSize]float32) {
	h0, h1, h2 := &hist[0], &hist[1], &hist[2]
	c := &bsFilterCoeffs
	for k := 0; k < SplitBandSize; k++ {
		w0 := (*[16]float32)(h0[k : k+16])
		w1 := (*[16]float32)(h1[k : k+16])
		w2 := (*[16]float32)(h2[k : k+16])

		// Phase 0: filters 0 {2,2,2} and 5 {-2,-2,-2}.
		t := float32(2 * bsTap(w0, 0, &c[0]))
		o0 := float32(0) + t
		o1 := float32(0) + t
		o2 := float32(0) + t
		t = float32(2 * bsTap(w0, 2, &c[5]))
		o0 -= t
		o1 -= t
		o2 -= t

		// Phase 1: filters 1 {s,0,-s}, 3 {-1,2,-1}, 6 {-s,0,s}, 8 {1,-2,1}.
		t = float32(bsSqrt3 * bsTap(w1, 0, &c[1]))
		o0 += t
		o2 -= t
		v := bsTap(w1, 1, &c[3])
		t = float32(2 * v)
		o0 -= v
		o1 += t
		o2 -= v
		t = float32(bsSqrt3 * bsTap(w1, 2, &c[6]))
		o0 -= t
		o2 += t
		v = bsTap(w1, 3, &c[8])
		t = float32(2 * v)
		o0 += v
		o1 -= t
		o2 += v

		// Phase 2: filters 2 {1,-2,1}, 4 {-s,0,s}, 7 {-1,2,-1}, 9 {s,0,-s}.
		v = bsTap(w2, 0, &c[2])
		t = float32(2 * v)
		o0 += v
		o1 -= t
		o2 += v
		t = float32(bsSqrt3 * bsTap(w2, 1, &c[4]))
		o0 -= t
		o2 += t
		v = bsTap(w2, 2, &c[7])
		t = float32(2 * v)
		o0 -= v
		o1 += t
		o2 -= v
		t = float32(bsSqrt3 * bsTap(w2, 3, &c[9]))
		o0 += t
		o2 -= t

		out0[k] = o0
		out1[k] = o1
		out2[k] = o2
	}
}

// bsAnalysis2Go is bsAnalysis1Go with two output samples per step: a and o
// are the sample at k, b and p the one at k+1.
func bsAnalysis2Go(hist *[bsSubSampling][bsHistSize]float32, out0, out1, out2 *[SplitBandSize]float32) {
	h0, h1, h2 := &hist[0], &hist[1], &hist[2]
	c := &bsFilterCoeffs
	for k := 0; k < SplitBandSize; k += 2 {
		w0 := (*[17]float32)(h0[k : k+17])
		w1 := (*[17]float32)(h1[k : k+17])
		w2 := (*[17]float32)(h2[k : k+17])

		// Phase 0: filters 0 {2,2,2} and 5 {-2,-2,-2}.
		a, b := bsTapAt(w0, 15, &c[0]), bsTapAt(w0, 16, &c[0])
		t, u := float32(2*a), float32(2*b)
		o0, p0 := float32(0)+t, float32(0)+u
		o1, p1 := float32(0)+t, float32(0)+u
		o2, p2 := float32(0)+t, float32(0)+u
		a, b = bsTapAt(w0, 13, &c[5]), bsTapAt(w0, 14, &c[5])
		t, u = float32(2*a), float32(2*b)
		o0, p0 = o0-t, p0-u
		o1, p1 = o1-t, p1-u
		o2, p2 = o2-t, p2-u

		// Phase 1: filters 1 {s,0,-s}, 3 {-1,2,-1}, 6 {-s,0,s}, 8 {1,-2,1}.
		a, b = bsTapAt(w1, 15, &c[1]), bsTapAt(w1, 16, &c[1])
		t, u = float32(bsSqrt3*a), float32(bsSqrt3*b)
		o0, p0 = o0+t, p0+u
		o2, p2 = o2-t, p2-u
		a, b = bsTapAt(w1, 14, &c[3]), bsTapAt(w1, 15, &c[3])
		t, u = float32(2*a), float32(2*b)
		o0, p0 = o0-a, p0-b
		o1, p1 = o1+t, p1+u
		o2, p2 = o2-a, p2-b
		a, b = bsTapAt(w1, 13, &c[6]), bsTapAt(w1, 14, &c[6])
		t, u = float32(bsSqrt3*a), float32(bsSqrt3*b)
		o0, p0 = o0-t, p0-u
		o2, p2 = o2+t, p2+u
		a, b = bsTapAt(w1, 12, &c[8]), bsTapAt(w1, 13, &c[8])
		t, u = float32(2*a), float32(2*b)
		o0, p0 = o0+a, p0+b
		o1, p1 = o1-t, p1-u
		o2, p2 = o2+a, p2+b

		// Phase 2: filters 2 {1,-2,1}, 4 {-s,0,s}, 7 {-1,2,-1}, 9 {s,0,-s}.
		a, b = bsTapAt(w2, 15, &c[2]), bsTapAt(w2, 16, &c[2])
		t, u = float32(2*a), float32(2*b)
		o0, p0 = o0+a, p0+b
		o1, p1 = o1-t, p1-u
		o2, p2 = o2+a, p2+b
		a, b = bsTapAt(w2, 14, &c[4]), bsTapAt(w2, 15, &c[4])
		t, u = float32(bsSqrt3*a), float32(bsSqrt3*b)
		o0, p0 = o0-t, p0-u
		o2, p2 = o2+t, p2+u
		a, b = bsTapAt(w2, 13, &c[7]), bsTapAt(w2, 14, &c[7])
		t, u = float32(2*a), float32(2*b)
		o0, p0 = o0-a, p0-b
		o1, p1 = o1+t, p1+u
		o2, p2 = o2-a, p2-b
		a, b = bsTapAt(w2, 12, &c[9]), bsTapAt(w2, 13, &c[9])
		t, u = float32(bsSqrt3*a), float32(bsSqrt3*b)
		o0, p0 = o0+t, p0+u
		o2, p2 = o2-t, p2-u

		y0 := (*[2]float32)(out0[k : k+2])
		y1 := (*[2]float32)(out1[k : k+2])
		y2 := (*[2]float32)(out2[k : k+2])
		y0[0], y0[1] = o0, p0
		y1[0], y1[1] = o1, p1
		y2[0], y2[1] = o2, p2
	}
}

func bsSynthesisGo(hist *[bsNumSynthesisInputs][bsHistSize]float32, in0, in1, in2 *[SplitBandSize]float32,
	out *[FullBandSize]float32) {
	// Modulate: the filter input is ((0 + m0*b0) + m1*b1) + m2*b2.
	g0, g1, g2, g3, g4, g5 := &hist[0], &hist[1], &hist[2], &hist[3], &hist[4], &hist[5]
	for k := 0; k < SplitBandSize; k++ {
		b0, b1, b2 := in0[k], in1[k], in2[k]
		p0, p1, p2 := float32(2*b0), float32(2*b1), float32(2*b2)
		q0, q2 := float32(bsSqrt3*b0), float32(bsSqrt3*b2)
		g0[MemorySize+k] = float32(0) + p0 + p1 + p2
		g1[MemorySize+k] = float32(0) - p0 - p1 - p2
		g2[MemorySize+k] = float32(0) + q0 - q2
		g3[MemorySize+k] = float32(0) - q0 + q2
		g4[MemorySize+k] = float32(0) + b0 - p1 + b2
		g5[MemorySize+k] = float32(0) - b0 + p1 - b2
	}

	// Filter, scale and upsample: phase d lands at out[d + bsSubSampling*k]
	// as the sum of upsamplingScaling * v over the phase's filters in inShift
	// order, starting from zero.
	if bsIsAMD64 {
		bsSynthesisFilter1Go(hist, out)
	} else {
		bsSynthesisFilter2Go(hist, out)
	}

	for i := range hist {
		copy(hist[i][:MemorySize], hist[i][SplitBandSize:])
	}
}

// bsSynthesisFilter1Go filters, scales and upsamples one sample of each
// phase per step.
func bsSynthesisFilter1Go(hist *[bsNumSynthesisInputs][bsHistSize]float32, out *[FullBandSize]float32) {
	g0, g1, g2, g3, g4, g5 := &hist[0], &hist[1], &hist[2], &hist[3], &hist[4], &hist[5]
	c := &bsFilterCoeffs
	const scale float32 = bsSubSampling
	for k := 0; k < SplitBandSize; k++ {
		w0 := (*[16]float32)(g0[k : k+16])
		w1 := (*[16]float32)(g1[k : k+16])
		w2 := (*[16]float32)(g2[k : k+16])
		w3 := (*[16]float32)(g3[k : k+16])
		w4 := (*[16]float32)(g4[k : k+16])
		w5 := (*[16]float32)(g5[k : k+16])
		y := (*[3]float32)(out[3*k : 3*k+3])

		// Phase 0: filters 0 and 5.
		o := float32(0) + float32(scale*bsTap(w0, 0, &c[0]))
		o += float32(scale * bsTap(w1, 2, &c[5]))
		y[0] = o

		// Phase 1: filters 1, 3, 6 and 8.
		o = float32(0) + float32(scale*bsTap(w2, 0, &c[1]))
		o += float32(scale * bsTap(w5, 1, &c[3]))
		o += float32(scale * bsTap(w3, 2, &c[6]))
		o += float32(scale * bsTap(w4, 3, &c[8]))
		y[1] = o

		// Phase 2: filters 2, 4, 7 and 9.
		o = float32(0) + float32(scale*bsTap(w4, 0, &c[2]))
		o += float32(scale * bsTap(w3, 1, &c[4]))
		o += float32(scale * bsTap(w5, 2, &c[7]))
		o += float32(scale * bsTap(w2, 3, &c[9]))
		y[2] = o
	}
}

// bsSynthesisFilter2Go is bsSynthesisFilter1Go with two samples of each
// phase per step: a and o are the sample at k, b and p the one at k+1.
func bsSynthesisFilter2Go(hist *[bsNumSynthesisInputs][bsHistSize]float32, out *[FullBandSize]float32) {
	g0, g1, g2, g3, g4, g5 := &hist[0], &hist[1], &hist[2], &hist[3], &hist[4], &hist[5]
	c := &bsFilterCoeffs
	const scale float32 = bsSubSampling
	for k := 0; k < SplitBandSize; k += 2 {
		w0 := (*[17]float32)(g0[k : k+17])
		w1 := (*[17]float32)(g1[k : k+17])
		w2 := (*[17]float32)(g2[k : k+17])
		w3 := (*[17]float32)(g3[k : k+17])
		w4 := (*[17]float32)(g4[k : k+17])
		w5 := (*[17]float32)(g5[k : k+17])
		y := (*[6]float32)(out[3*k : 3*k+6])

		// Phase 0: filters 0 and 5.
		a, b := bsTapAt(w0, 15, &c[0]), bsTapAt(w0, 16, &c[0])
		o, p := float32(0)+float32(scale*a), float32(0)+float32(scale*b)
		a, b = bsTapAt(w1, 13, &c[5]), bsTapAt(w1, 14, &c[5])
		o, p = o+float32(scale*a), p+float32(scale*b)
		y[0], y[3] = o, p

		// Phase 1: filters 1, 3, 6 and 8.
		a, b = bsTapAt(w2, 15, &c[1]), bsTapAt(w2, 16, &c[1])
		o, p = float32(0)+float32(scale*a), float32(0)+float32(scale*b)
		a, b = bsTapAt(w5, 14, &c[3]), bsTapAt(w5, 15, &c[3])
		o, p = o+float32(scale*a), p+float32(scale*b)
		a, b = bsTapAt(w3, 13, &c[6]), bsTapAt(w3, 14, &c[6])
		o, p = o+float32(scale*a), p+float32(scale*b)
		a, b = bsTapAt(w4, 12, &c[8]), bsTapAt(w4, 13, &c[8])
		o, p = o+float32(scale*a), p+float32(scale*b)
		y[1], y[4] = o, p

		// Phase 2: filters 2, 4, 7 and 9.
		a, b = bsTapAt(w4, 15, &c[2]), bsTapAt(w4, 16, &c[2])
		o, p = float32(0)+float32(scale*a), float32(0)+float32(scale*b)
		a, b = bsTapAt(w3, 14, &c[4]), bsTapAt(w3, 15, &c[4])
		o, p = o+float32(scale*a), p+float32(scale*b)
		a, b = bsTapAt(w5, 13, &c[7]), bsTapAt(w5, 14, &c[7])
		o, p = o+float32(scale*a), p+float32(scale*b)
		a, b = bsTapAt(w2, 12, &c[9]), bsTapAt(w2, 13, &c[9])
		o, p = o+float32(scale*a), p+float32(scale*b)
		y[2], y[5] = o, p
	}
}
