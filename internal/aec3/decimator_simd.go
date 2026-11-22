//go:build goexperiment.simd && (arm64 || amd64) && !purego

package aec3

import "simd/archsimd"

// dlyHasDecimateKernel reports whether dlyDecimateDs4 is an archsimd kernel.
// dlyDs4HasSimd reports whether the CPU runs it.
const dlyHasDecimateKernel = true

// dlyDs4Skew is the number of samples a section of the kernel runs behind
// the one before it.
const dlyDs4Skew = 2

// dlyDs4Coef holds the coefficients B[0], B[1], B[2], A[0], A[1] of the
// decimation by 4, one vector each with one lane per section: the three
// anti-aliasing sections, then the noise reduction one.
var dlyDs4Coef = func() (c [5][4]float32) {
	for s := range c[0] {
		q := &dlyHighPassFilter[0]
		if s < len(dlyLowPassFilterDs4) {
			q = &dlyLowPassFilterDs4[s]
		}
		c[0][s], c[1][s], c[2][s], c[3][s], c[4][s] = q.B[0], q.B[1], q.B[2], q.A[0], q.A[1]
	}
	return
}()

// dlyDecimateDs4 applies the four sections of s to in and writes every
// fourth output sample to out. It reports false and leaves s alone when a
// value of the block is not finite or the CPU lacks the instructions.
//
// Lane i is section i. At step t it processes sample t - 2i, whose input is
// the output of section i-1 two steps before: y[1] of that lane. A lane
// performs the operations of dlyApplyBiQuad in their order, so the result is
// that of the scalar sections. The skew of two samples keeps the path
// through the input of a lane off the critical cycle, which runs from an
// output of a section to its next one.
//
// The samples the lanes do not cover (the first 2i of section i < 3, the
// last 2i of section i > 0) run through dlyApplyBiQuad.
//
// A vector addition may take its operands in the other order than the scalar
// one, which shows only in the payload of a NaN from two NaNs. An output
// that is not finite keeps every later output of its section from being
// finite, so the final states tell whether every value was finite.
func dlyDecimateDs4(in *[BlockSize]float32, out *[BlockSize / 4]float32, s *[4]BiQuad) bool {
	const skew = dlyDs4Skew
	if !dlyDs4HasSimd {
		return false
	}
	saved := *s

	// The samples before the first step: 6 of section 0, 4 of section 1, 2
	// of section 2.
	var pa [3 * skew]float32
	var pb [2 * skew]float32
	var pc [skew]float32
	dlyApplyBiQuad(in[:3*skew], pa[:], &s[0])
	dlyApplyBiQuad(pa[:2*skew], pb[:], &s[1])
	dlyApplyBiQuad(pb[:skew], pc[:], &s[2])

	var x0, x1, y0, y1 archsimd.Float32x4
	x0 = x0.SetElem(0, s[0].X[0]).SetElem(1, s[1].X[0]).SetElem(2, s[2].X[0]).SetElem(3, s[3].X[0])
	x1 = x1.SetElem(0, s[0].X[1]).SetElem(1, s[1].X[1]).SetElem(2, s[2].X[1]).SetElem(3, s[3].X[1])
	y0 = y0.SetElem(0, s[0].Y[0]).SetElem(1, s[1].Y[0]).SetElem(2, s[2].Y[0]).SetElem(3, s[3].Y[0])
	y1 = y1.SetElem(0, s[0].Y[1]).SetElem(1, s[1].Y[1]).SetElem(2, s[2].Y[1]).SetElem(3, s[3].Y[1])
	b0 := archsimd.LoadFloat32x4Array(&dlyDs4Coef[0])
	b1 := archsimd.LoadFloat32x4Array(&dlyDs4Coef[1])
	b2 := archsimd.LoadFloat32x4Array(&dlyDs4Coef[2])
	a0 := archsimd.LoadFloat32x4Array(&dlyDs4Coef[3])
	a1 := archsimd.LoadFloat32x4Array(&dlyDs4Coef[4])

	// Step t = j + 3 for t in [6, 64): lane 3 of the load is in[t] and lane
	// 3 of the result is output sample t - 6. A pass is four steps, the
	// first of which yields a sample of out.
	for j := 3; j < BlockSize-8; j += 4 {
		p := (*[7]float32)(in[j : j+7])
		va, wa := dlyDs4Step((*[4]float32)(p[0:4]), x0, x1, y0, y1, b0, b1, b2, a0, a1)
		vb, wb := dlyDs4Step((*[4]float32)(p[1:5]), va, x0, wa, y0, b0, b1, b2, a0, a1)
		vc, wc := dlyDs4Step((*[4]float32)(p[2:6]), vb, va, wb, wa, b0, b1, b2, a0, a1)
		vd, wd := dlyDs4Step((*[4]float32)(p[3:7]), vc, vb, wc, wb, b0, b1, b2, a0, a1)
		x1, x0, y1, y0 = vc, vd, wc, wd
		out[((j-3)>>2)&15] = wa.GetElem(3)
	}
	{
		// Steps 62 and 63: output samples 56 and 57.
		va, wa := dlyDs4Step((*[4]float32)(in[BlockSize-5:BlockSize-1]), x0, x1, y0, y1, b0, b1, b2, a0, a1)
		vb, wb := dlyDs4Step((*[4]float32)(in[BlockSize-4:BlockSize]), va, x0, wa, y0, b0, b1, b2, a0, a1)
		x1, x0, y1, y0 = va, vb, wa, wb
		out[14] = wa.GetElem(3)
	}

	var tx0, tx1, ty0, ty1 [4]float32
	x0.StoreArray(&tx0)
	x1.StoreArray(&tx1)
	y0.StoreArray(&ty0)
	y1.StoreArray(&ty1)
	if f := (ty0[0] - ty0[0]) + (ty0[1] - ty0[1]) + (ty0[2] - ty0[2]) + (ty0[3] - ty0[3]); f != 0 {
		*s = saved
		return false
	}
	for i := range s {
		s[i].X[0], s[i].X[1], s[i].Y[0], s[i].Y[1] = tx0[i], tx1[i], ty0[i], ty1[i]
	}

	// The samples after the last step: 2 of section 1, 4 of section 2, 6 of
	// section 3. The last two outputs of a section are its y state.
	qa := [skew]float32{ty1[0], ty0[0]}
	qb := [2 * skew]float32{ty1[1], ty0[1]}
	qc := [3 * skew]float32{ty1[2], ty0[2]}
	var qy [3 * skew]float32
	dlyApplyBiQuad(qa[:], qb[skew:], &s[1])
	dlyApplyBiQuad(qb[:], qc[skew:], &s[2])
	dlyApplyBiQuad(qc[:], qy[:], &s[3])
	// qy is output samples 58..63.
	out[15] = qy[2]
	return true
}

// dlyDs4Step is one step of dlyDecimateDs4. p ends with the input sample of
// section 0; x0, x1, y0 and y1 are the states of the lanes. It returns the
// inputs v and the outputs w of the lanes.
func dlyDs4Step(p *[4]float32, x0, x1, y0, y1, b0, b1, b2, a0, a1 archsimd.Float32x4) (v, w archsimd.Float32x4) {
	// {p[3], y1[0], y1[1], y1[2]} (EXT #12, PALIGNR $12).
	xin := archsimd.LoadFloat32x4Array(p).ToBits().ReshapeToUint8s()
	v = y1.ToBits().ReshapeToUint8s().ConcatShiftBytesRight(xin, 12).ReshapeToUint32s().BitsToFloat32()
	w = b0.Mul(v).Add(b1.Mul(x0)).Add(b2.Mul(x1)).Sub(a0.Mul(y0)).Sub(a1.Mul(y1))
	return
}
