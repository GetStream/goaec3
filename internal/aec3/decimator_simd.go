//go:build goexperiment.simd && (arm64 || amd64) && !purego

package aec3

import "simd/archsimd"

// dlyHasDecimateKernel reports that dlyDecimateDs4 is an archsimd kernel;
// dlyDs4HasSimd (per architecture) reports that the CPU runs it.
const dlyHasDecimateKernel = true

// dlyDs4Skew is the number of samples a section of the kernel runs behind
// the section before it.
const dlyDs4Skew = 2

// dlyDs4Coef holds the coefficients of the four sections of the decimation
// by 4 (the three anti-aliasing sections, then the noise reduction one), one
// vector per coefficient with one lane per section: B[0], B[1], B[2], A[0],
// A[1].
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

// dlyDecimateDs4 is the decimation by 4 of one block: the four sections of
// s applied to in, and every fourth output sample written to out. It
// reports false, with s as it was, when a value of the block is not finite
// or the CPU lacks the vector instructions; the caller then runs the scalar
// sections.
//
// A section computes, per sample, five products and a chain of four
// additions, and the scalar code runs the 36 operations of the four
// sections one sample after the other. The kernel runs the four sections as
// the four lanes of one vector: lane i is section i, and at step t it
// processes sample t - 2i, whose input is the output of section i-1 two
// steps before, which is that lane's y[1] state. Every lane performs the
// operations of dlyApplyBiQuad in the same order on the same values, so the
// result is the same; a step is ten vector operations, and the two samples
// between the sections keep the path through the input (a shuffle, a
// product and four additions) off the critical cycle, which is the path
// from an output to the next one of the same section.
//
// The samples the lanes do not cover (the first 2i of section i < 3 before
// the first step, the last 2i of section i > 0 after the last one) run
// through dlyApplyBiQuad.
//
// The scalar additions and the vector ones may differ in operand order,
// which only matters when both operands are NaN. An output that is not
// finite keeps every later output of its section from being finite, so the
// states after the last step tell whether every value of the steps was
// finite.
func dlyDecimateDs4(in *[BlockSize]float32, out *[BlockSize / 4]float32, s *[4]BiQuad) bool {
	const skew = dlyDs4Skew
	if !dlyDs4HasSimd {
		return false
	}
	saved := *s

	// The samples before the first step: 6 of section 0, 4 of section 1 and
	// 2 of section 2.
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

	// Step t = j + 3 for t in [6, 64): lane 3 of the load is in[t], and lane
	// 3 of the result is output sample t - 6 = j - 3. Four steps per pass,
	// the first of which gives an output sample of the decimation; the
	// states change names instead of registers.
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

	// The samples after the last step: 2 of section 1, 4 of section 2 and 6
	// of section 3. The two outputs a section produced last are its y state.
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

// dlyDs4Step is one step of dlyDecimateDs4: p ends with the input sample of
// section 0, x0, x1, y0 and y1 are the states of the lanes, and v and w are
// the inputs and the outputs of the lanes.
func dlyDs4Step(p *[4]float32, x0, x1, y0, y1, b0, b1, b2, a0, a1 archsimd.Float32x4) (v, w archsimd.Float32x4) {
	// {p[3], y1[0], y1[1], y1[2]} (EXT #12, PALIGNR $12).
	xin := archsimd.LoadFloat32x4Array(p).ToBits().ReshapeToUint8s()
	v = y1.ToBits().ReshapeToUint8s().ConcatShiftBytesRight(xin, 12).ReshapeToUint32s().BitsToFloat32()
	w = b0.Mul(v).Add(b1.Mul(x0)).Add(b2.Mul(x1)).Sub(a0.Mul(y0)).Sub(a1.Mul(y1))
	return
}
