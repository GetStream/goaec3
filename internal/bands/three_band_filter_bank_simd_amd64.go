//go:build goexperiment.simd && amd64 && !purego

package bands

import "simd/archsimd"

// The 256-bit ThreeBandFilterBank kernels: those of
// three_band_filter_bank_simd.go with eight output samples per step.

var bsHasAVX2 = archsimd.X86.AVX2()

func bsAnalysisKernel(hist *[bsSubSampling][bsHistSize]float32, in *[FullBandSize]float32,
	out0, out1, out2 *[SplitBandSize]float32) {
	if bsHasAVX2 {
		bsAnalysisAVX2(hist, in, out0, out1, out2)
		return
	}
	bsAnalysisSIMD(hist, in, out0, out1, out2)
}

func bsSynthesisKernel(hist *[bsNumSynthesisInputs][bsHistSize]float32,
	in0, in1, in2 *[SplitBandSize]float32, out *[FullBandSize]float32) {
	if bsHasAVX2 {
		bsSynthesisAVX2(hist, in0, in1, in2, out)
		return
	}
	bsSynthesisSIMD(hist, in0, in1, in2, out)
}

// bsCoeffs8 holds the coefficients of one filter broadcast to 256 bits.
type bsCoeffs8 struct{ c0, c1, c2, c3 archsimd.Float32x8 }

// bsBroadcastCoeffs8 fills c in place: a returned struct is zeroed with
// 128-bit stores between the 256-bit broadcasts.
func bsBroadcastCoeffs8(c *[NumNonZeroFilters]bsCoeffs8) {
	for i := range c {
		f := &bsFilterCoeffs[i]
		c[i].c0 = archsimd.BroadcastFloat32x8(f[0])
		c[i].c1 = archsimd.BroadcastFloat32x8(f[1])
		c[i].c2 = archsimd.BroadcastFloat32x8(f[2])
		c[i].c3 = archsimd.BroadcastFloat32x8(f[3])
	}
}

// bsTap8 is bsFilterCore for out[k:k+8]: w is the history window
// h[k : k+23] and c the coefficients of a filter with shift inShift.
func bsTap8(w *[23]float32, inShift int, c *bsCoeffs8) archsimd.Float32x8 {
	var acc archsimd.Float32x8
	acc = acc.Add(archsimd.LoadFloat32x8Array((*[8]float32)(w[15-inShift : 23-inShift])).Mul(c.c0))
	acc = acc.Add(archsimd.LoadFloat32x8Array((*[8]float32)(w[11-inShift : 19-inShift])).Mul(c.c1))
	acc = acc.Add(archsimd.LoadFloat32x8Array((*[8]float32)(w[7-inShift : 15-inShift])).Mul(c.c2))
	return acc.Add(archsimd.LoadFloat32x8Array((*[8]float32)(w[3-inShift : 11-inShift])).Mul(c.c3))
}

// bsAnalysisAVX2 is bsAnalysisSIMD with eight output samples per step.
func bsAnalysisAVX2(hist *[bsSubSampling][bsHistSize]float32, in *[FullBandSize]float32,
	out0, out1, out2 *[SplitBandSize]float32) {
	// Downsample: phase d takes in[(bsSubSampling-1) - d + bsSubSampling*k].
	h0, h1, h2 := &hist[0], &hist[1], &hist[2]
	for k := 0; k < SplitBandSize; k++ {
		x := (*[3]float32)(in[3*k : 3*k+3])
		h0[MemorySize+k] = x[2]
		h1[MemorySize+k] = x[1]
		h2[MemorySize+k] = x[0]
	}

	var c [NumNonZeroFilters]bsCoeffs8
	bsBroadcastCoeffs8(&c)
	two := archsimd.BroadcastFloat32x8(2)
	sqrt3 := archsimd.BroadcastFloat32x8(bsSqrt3)
	var zero archsimd.Float32x8
	for k := 0; k < SplitBandSize; k += 8 {
		w0 := (*[23]float32)(h0[k : k+23])
		w1 := (*[23]float32)(h1[k : k+23])
		w2 := (*[23]float32)(h2[k : k+23])

		// Phase 0: filters 0 {2,2,2} and 5 {-2,-2,-2}.
		t := two.Mul(bsTap8(w0, 0, &c[0]))
		o0 := zero.Add(t)
		o1 := o0
		o2 := o0
		t = two.Mul(bsTap8(w0, 2, &c[5]))
		o0 = o0.Sub(t)
		o1 = o1.Sub(t)
		o2 = o2.Sub(t)

		// Phase 1: filters 1 {s,0,-s}, 3 {-1,2,-1}, 6 {-s,0,s}, 8 {1,-2,1}.
		t = sqrt3.Mul(bsTap8(w1, 0, &c[1]))
		o0 = o0.Add(t)
		o2 = o2.Sub(t)
		v := bsTap8(w1, 1, &c[3])
		o0 = o0.Sub(v)
		o1 = o1.Add(two.Mul(v))
		o2 = o2.Sub(v)
		t = sqrt3.Mul(bsTap8(w1, 2, &c[6]))
		o0 = o0.Sub(t)
		o2 = o2.Add(t)
		v = bsTap8(w1, 3, &c[8])
		o0 = o0.Add(v)
		o1 = o1.Sub(two.Mul(v))
		o2 = o2.Add(v)

		// Phase 2: filters 2 {1,-2,1}, 4 {-s,0,s}, 7 {-1,2,-1}, 9 {s,0,-s}.
		v = bsTap8(w2, 0, &c[2])
		o0 = o0.Add(v)
		o1 = o1.Sub(two.Mul(v))
		o2 = o2.Add(v)
		t = sqrt3.Mul(bsTap8(w2, 1, &c[4]))
		o0 = o0.Sub(t)
		o2 = o2.Add(t)
		v = bsTap8(w2, 2, &c[7])
		o0 = o0.Sub(v)
		o1 = o1.Add(two.Mul(v))
		o2 = o2.Sub(v)
		t = sqrt3.Mul(bsTap8(w2, 3, &c[9]))
		o0 = o0.Add(t)
		o2 = o2.Sub(t)

		o0.StoreArray((*[8]float32)(out0[k : k+8]))
		o1.StoreArray((*[8]float32)(out1[k : k+8]))
		o2.StoreArray((*[8]float32)(out2[k : k+8]))
	}
	archsimd.ClearAVXUpperBits()

	for d := range hist {
		copy(hist[d][:MemorySize], hist[d][SplitBandSize:])
	}
}

// bsSynthesisAVX2 is bsSynthesisSIMD with eight output samples per step.
// The interleave into out moves single floats, which must not run with a
// dirty upper register half: the phases are filtered into t and interleaved
// after ClearAVXUpperBits.
func bsSynthesisAVX2(hist *[bsNumSynthesisInputs][bsHistSize]float32,
	in0, in1, in2 *[SplitBandSize]float32, out *[FullBandSize]float32) {
	var t [bsSubSampling][SplitBandSize]float32
	bsSynthesisFilterAVX2(hist, in0, in1, in2, &t)
	t0, t1, t2 := &t[0], &t[1], &t[2]
	for k := 0; k < SplitBandSize; k += 4 {
		y := (*[12]float32)(out[3*k : 3*k+12])
		a := (*[4]float32)(t0[k : k+4])
		b := (*[4]float32)(t1[k : k+4])
		c := (*[4]float32)(t2[k : k+4])
		y[0], y[1], y[2] = a[0], b[0], c[0]
		y[3], y[4], y[5] = a[1], b[1], c[1]
		y[6], y[7], y[8] = a[2], b[2], c[2]
		y[9], y[10], y[11] = a[3], b[3], c[3]
	}

	for i := range hist {
		copy(hist[i][:MemorySize], hist[i][SplitBandSize:])
	}
}

// bsSynthesisFilterAVX2 modulates the bands into hist and filters and scales
// the three output phases into t.
func bsSynthesisFilterAVX2(hist *[bsNumSynthesisInputs][bsHistSize]float32,
	in0, in1, in2 *[SplitBandSize]float32, t *[bsSubSampling][SplitBandSize]float32) {
	// The coefficient array is zeroed with 128-bit stores, so it is set up
	// before the first 256-bit operation.
	var c [NumNonZeroFilters]bsCoeffs8
	bsBroadcastCoeffs8(&c)

	// Modulate: the filter input is ((0 + m0*b0) + m1*b1) + m2*b2.
	g0, g1, g2, g3, g4, g5 := &hist[0], &hist[1], &hist[2], &hist[3], &hist[4], &hist[5]
	two := archsimd.BroadcastFloat32x8(2)
	sqrt3 := archsimd.BroadcastFloat32x8(bsSqrt3)
	var zero archsimd.Float32x8
	for k := 0; k < SplitBandSize; k += 8 {
		b0 := archsimd.LoadFloat32x8Array((*[8]float32)(in0[k : k+8]))
		b1 := archsimd.LoadFloat32x8Array((*[8]float32)(in1[k : k+8]))
		b2 := archsimd.LoadFloat32x8Array((*[8]float32)(in2[k : k+8]))
		p0, p1, p2 := two.Mul(b0), two.Mul(b1), two.Mul(b2)
		q0, q2 := sqrt3.Mul(b0), sqrt3.Mul(b2)
		zero.Add(p0).Add(p1).Add(p2).StoreArray((*[8]float32)(g0[MemorySize+k : MemorySize+k+8]))
		zero.Sub(p0).Sub(p1).Sub(p2).StoreArray((*[8]float32)(g1[MemorySize+k : MemorySize+k+8]))
		zero.Add(q0).Sub(q2).StoreArray((*[8]float32)(g2[MemorySize+k : MemorySize+k+8]))
		zero.Sub(q0).Add(q2).StoreArray((*[8]float32)(g3[MemorySize+k : MemorySize+k+8]))
		zero.Add(b0).Sub(p1).Add(b2).StoreArray((*[8]float32)(g4[MemorySize+k : MemorySize+k+8]))
		zero.Sub(b0).Add(p1).Sub(b2).StoreArray((*[8]float32)(g5[MemorySize+k : MemorySize+k+8]))
	}

	// Filter, scale and upsample: phase d lands at out[d + bsSubSampling*k]
	// as the sum of upsamplingScaling * v over the phase's filters in inShift
	// order, starting from zero.
	scale := archsimd.BroadcastFloat32x8(bsSubSampling)
	t0, t1, t2 := &t[0], &t[1], &t[2]
	for k := 0; k < SplitBandSize; k += 8 {
		w0 := (*[23]float32)(g0[k : k+23])
		w1 := (*[23]float32)(g1[k : k+23])
		w2 := (*[23]float32)(g2[k : k+23])
		w3 := (*[23]float32)(g3[k : k+23])
		w4 := (*[23]float32)(g4[k : k+23])
		w5 := (*[23]float32)(g5[k : k+23])

		// Phase 0: filters 0 and 5.
		o0 := zero.Add(scale.Mul(bsTap8(w0, 0, &c[0])))
		o0 = o0.Add(scale.Mul(bsTap8(w1, 2, &c[5])))

		// Phase 1: filters 1, 3, 6 and 8.
		o1 := zero.Add(scale.Mul(bsTap8(w2, 0, &c[1])))
		o1 = o1.Add(scale.Mul(bsTap8(w5, 1, &c[3])))
		o1 = o1.Add(scale.Mul(bsTap8(w3, 2, &c[6])))
		o1 = o1.Add(scale.Mul(bsTap8(w4, 3, &c[8])))

		// Phase 2: filters 2, 4, 7 and 9.
		o2 := zero.Add(scale.Mul(bsTap8(w4, 0, &c[2])))
		o2 = o2.Add(scale.Mul(bsTap8(w3, 1, &c[4])))
		o2 = o2.Add(scale.Mul(bsTap8(w5, 2, &c[7])))
		o2 = o2.Add(scale.Mul(bsTap8(w2, 3, &c[9])))

		o0.StoreArray((*[8]float32)(t0[k : k+8]))
		o1.StoreArray((*[8]float32)(t1[k : k+8]))
		o2.StoreArray((*[8]float32)(t2[k : k+8]))
	}
	archsimd.ClearAVXUpperBits()
}
