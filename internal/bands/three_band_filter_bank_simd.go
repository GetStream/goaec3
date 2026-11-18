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

const bsHistSize = MemorySize + SplitBandSize

// bsSchedule lists the non-zero filters in the reference loop order
// (downsampling/upsampling index outer, inShift inner).
var bsSchedule = [NumNonZeroFilters]struct{ phase, inShift, filter int }{
	{0, 0, 0}, {0, 2, 5},
	{1, 0, 1}, {1, 1, 3}, {1, 2, 6}, {1, 3, 8},
	{2, 0, 2}, {2, 1, 4}, {2, 2, 7}, {2, 3, 9},
}

// ThreeBandFilterBank splits a 48 kHz frame into three 16 kHz bands and
// merges them back. The prototype low-pass filter has half bandwidth, so a
// DCT shifts it to the center frequencies [1/12, 3/12, 5/12].
type ThreeBandFilterBank struct {
	// analysisHist[phase] is the filter input for downsampling index phase.
	analysisHist [bsSubSampling][bsHistSize]float32
	// synthesisHist[filter] is the modulated input of filter.
	synthesisHist [NumNonZeroFilters][bsHistSize]float32
	// synthesisOut[phase] is the upsampled output at phase, before
	// interleaving.
	synthesisOut [bsSubSampling][SplitBandSize]float32

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
	bsAnalysisSIMD(&f.analysisHist, in, (*[SplitBandSize]float32)(out[0]),
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
	bsSynthesisSIMD(&f.synthesisHist, &f.synthesisOut, in0, in1, in2, out)
}

// synthesisRef runs the reference on the state held in synthesisHist.
func (f *ThreeBandFilterBank) synthesisRef(in *[NumBands][]float32, out *[FullBandSize]float32) {
	for fi := range f.synthesisHist {
		copy(f.ref.stateSynthesis[fi][:], f.synthesisHist[fi][:MemorySize])
	}
	f.ref.synthesis(in, out)
	for fi := range f.synthesisHist {
		copy(f.synthesisHist[fi][:MemorySize], f.ref.stateSynthesis[fi][:])
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
func bsAnalysisFinite(hist *[bsSubSampling][bsHistSize]float32, in *[FullBandSize]float32) bool {
	var acc archsimd.Float32x4
	for d := range hist {
		acc = bsFiniteState(acc, &hist[d])
	}
	for k := 0; k < FullBandSize; k += 4 {
		acc = bsFinite4(acc, (*[4]float32)(in[k:k+4]))
	}
	return bsFiniteResult(acc)
}

// bsSynthesisFinite reports whether the synthesis input and state are
// finite.
func bsSynthesisFinite(hist *[NumNonZeroFilters][bsHistSize]float32, in0, in1, in2 *[SplitBandSize]float32) bool {
	var acc archsimd.Float32x4
	for fi := range hist {
		acc = bsFiniteState(acc, &hist[fi])
	}
	for k := 0; k < SplitBandSize; k += 4 {
		acc = bsFinite4(acc, (*[4]float32)(in0[k:k+4]))
		acc = bsFinite4(acc, (*[4]float32)(in1[k:k+4]))
		acc = bsFinite4(acc, (*[4]float32)(in2[k:k+4]))
	}
	return bsFiniteResult(acc)
}

// bsCore4 is bsFilterCore for out[k:k+4], reading the history at
// b = MemorySize + k - inShift through the window w = h[b-12 : b+4].
func bsCore4(w *[16]float32, h0, h1, h2, h3 archsimd.Float32x4) archsimd.Float32x4 {
	var acc archsimd.Float32x4
	acc = acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[12:16])).Mul(h0))
	acc = acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[8:12])).Mul(h1))
	acc = acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[4:8])).Mul(h2))
	return acc.Add(archsimd.LoadFloat32x4Array((*[4]float32)(w[0:4])).Mul(h3))
}

// bsWindow returns the 16 history samples bsCore4 reads for out[k:k+4].
func bsWindow(h *[bsHistSize]float32, base, k int) *[16]float32 {
	b := base + k - 12
	return (*[16]float32)(h[b : b+16])
}

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

	for i, e := range bsSchedule {
		h := &hist[e.phase]
		c := &bsFilterCoeffs[e.filter]
		c0 := archsimd.BroadcastFloat32x4(c[0])
		c1 := archsimd.BroadcastFloat32x4(c[1])
		c2 := archsimd.BroadcastFloat32x4(c[2])
		c3 := archsimd.BroadcastFloat32x4(c[3])
		m := &bsDctModulation[e.filter]
		m0 := archsimd.BroadcastFloat32x4(m[0])
		m1 := archsimd.BroadcastFloat32x4(m[1])
		m2 := archsimd.BroadcastFloat32x4(m[2])
		base := MemorySize - e.inShift
		if i == 0 {
			// The bands start at zero: 0 + m*v.
			var zero archsimd.Float32x4
			for k := 0; k < SplitBandSize; k += 4 {
				v := bsCore4(bsWindow(h, base, k), c0, c1, c2, c3)
				zero.Add(m0.Mul(v)).StoreArray((*[4]float32)(out0[k : k+4]))
				zero.Add(m1.Mul(v)).StoreArray((*[4]float32)(out1[k : k+4]))
				zero.Add(m2.Mul(v)).StoreArray((*[4]float32)(out2[k : k+4]))
			}
			continue
		}
		for k := 0; k < SplitBandSize; k += 4 {
			v := bsCore4(bsWindow(h, base, k), c0, c1, c2, c3)
			o0 := (*[4]float32)(out0[k : k+4])
			o1 := (*[4]float32)(out1[k : k+4])
			o2 := (*[4]float32)(out2[k : k+4])
			archsimd.LoadFloat32x4Array(o0).Add(m0.Mul(v)).StoreArray(o0)
			archsimd.LoadFloat32x4Array(o1).Add(m1.Mul(v)).StoreArray(o1)
			archsimd.LoadFloat32x4Array(o2).Add(m2.Mul(v)).StoreArray(o2)
		}
	}

	for d := range hist {
		copy(hist[d][:MemorySize], hist[d][SplitBandSize:])
	}
}

func bsSynthesisSIMD(hist *[NumNonZeroFilters][bsHistSize]float32, tmp *[bsSubSampling][SplitBandSize]float32,
	in0, in1, in2 *[SplitBandSize]float32, out *[FullBandSize]float32) {
	// Modulate: filter input is ((0 + m0*b0) + m1*b1) + m2*b2.
	for fi := range hist {
		h := &hist[fi]
		m := &bsDctModulation[fi]
		m0 := archsimd.BroadcastFloat32x4(m[0])
		m1 := archsimd.BroadcastFloat32x4(m[1])
		m2 := archsimd.BroadcastFloat32x4(m[2])
		var zero archsimd.Float32x4
		for k := 0; k < SplitBandSize; k += 4 {
			x := zero.Add(m0.Mul(archsimd.LoadFloat32x4Array((*[4]float32)(in0[k : k+4]))))
			x = x.Add(m1.Mul(archsimd.LoadFloat32x4Array((*[4]float32)(in1[k : k+4]))))
			x = x.Add(m2.Mul(archsimd.LoadFloat32x4Array((*[4]float32)(in2[k : k+4]))))
			x.StoreArray((*[4]float32)(h[MemorySize+k : MemorySize+k+4]))
		}
	}

	// Filter and scale. Each phase's outputs start at zero and accumulate
	// upsamplingScaling * v over the phase's filters in inShift order.
	scale := archsimd.BroadcastFloat32x4(bsSubSampling)
	for i, e := range bsSchedule {
		h := &hist[e.filter]
		c := &bsFilterCoeffs[e.filter]
		c0 := archsimd.BroadcastFloat32x4(c[0])
		c1 := archsimd.BroadcastFloat32x4(c[1])
		c2 := archsimd.BroadcastFloat32x4(c[2])
		c3 := archsimd.BroadcastFloat32x4(c[3])
		base := MemorySize - e.inShift
		t := &tmp[e.phase]
		if i == 0 || bsSchedule[i-1].phase != e.phase {
			var zero archsimd.Float32x4
			for k := 0; k < SplitBandSize; k += 4 {
				v := bsCore4(bsWindow(h, base, k), c0, c1, c2, c3)
				zero.Add(scale.Mul(v)).StoreArray((*[4]float32)(t[k : k+4]))
			}
			continue
		}
		for k := 0; k < SplitBandSize; k += 4 {
			v := bsCore4(bsWindow(h, base, k), c0, c1, c2, c3)
			o := (*[4]float32)(t[k : k+4])
			archsimd.LoadFloat32x4Array(o).Add(scale.Mul(v)).StoreArray(o)
		}
	}

	// Upsample: phase d lands at out[d + bsSubSampling*k].
	t0, t1, t2 := &tmp[0], &tmp[1], &tmp[2]
	for k := 0; k < SplitBandSize; k++ {
		y := (*[3]float32)(out[3*k : 3*k+3])
		y[0] = t0[k]
		y[1] = t1[k]
		y[2] = t2[k]
	}

	for fi := range hist {
		copy(hist[fi][:MemorySize], hist[fi][SplitBandSize:])
	}
}
