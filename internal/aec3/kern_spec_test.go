package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// The references that are not a plain element loop are checked against the
// loop they stand for. Without archsimd the kernels are the references, and
// the kernel tests do not cover them.

func kernSumRowsSpec(dst *[FftLengthBy2Plus1]float32, rows []*[FftLengthBy2Plus1]float32) {
	*dst = [FftLengthBy2Plus1]float32{}
	for _, r := range rows {
		for k := range dst {
			dst[k] += r[k]
		}
	}
}

func kernUnpackSpec(v *[FftLength]float32, re, im *[FftLengthBy2Plus1]float32) {
	re[0] = v[0]
	re[FftLengthBy2] = v[1]
	im[0] = 0
	im[FftLengthBy2] = 0
	for k, j := 1, 2; k < FftLengthBy2; k, j = k+1, j+2 {
		re[k] = v[j]
		im[k] = v[j+1]
	}
}

func kernPackSpec(re, im *[FftLengthBy2Plus1]float32, v *[FftLength]float32) {
	v[0] = re[0]
	v[1] = re[FftLengthBy2]
	for k, j := 1, 2; k < FftLengthBy2; k, j = k+1, j+2 {
		v[j] = re[k]
		v[j+1] = im[k]
	}
}

func kernFir3Spec(x []float32, h *[3]float32, out []float32, from, to int) {
	for k := from; k <= to; k++ {
		var tmp float32
		for j := range h {
			tmp += float32(x[k-j] * h[j])
		}
		out[k] = tmp
	}
}

// kernRefinedLeakSpec is the H_error leakage loop of
// RefinedFilterUpdateGain::Compute as in the C++.
func kernRefinedLeakSpec(E2Refined, E2Coarse *[FftLengthBy2Plus1]float32, erl []float32, disallowLeakageDiverged bool,
	leakageConverged, leakageDiverged, errorFloor, errorCeil float32, hError *[FftLengthBy2Plus1]float32) {
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

func TestKernRefinedLeakMatchesSpec(t *testing.T) {
	rng := rand.New(rand.NewSource(43))
	for trial := 0; trial < 20000; trial++ {
		refined, coarse, erl := kernEdge65(rng), kernEdge65(rng), kernEdge65(rng)
		if trial%3 == 0 {
			// Equal and nearly equal powers, where <= decides.
			for k := range coarse {
				if rng.Intn(2) == 0 {
					coarse[k] = refined[k]
				}
			}
		}
		got := *kernEdge65(rng)
		want := got
		p := [4]float32{float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.Float64()), 1 + float32(rng.Float64())}
		if trial%5 == 0 {
			p = [4]float32{0.00005, 0.05, 0.001, 2}
		}
		dis := rng.Intn(3) == 0
		kernRefinedLeakGeneric(refined, coarse, erl[:], dis, p[0], p[1], p[2], p[3], &got)
		kernRefinedLeakSpec(refined, coarse, erl[:], dis, p[0], p[1], p[2], p[3], &want)
		kernCheckExact(t, "RefinedLeak", got[:], want[:])
	}
}

// kernSpectrumAvx2Spec is FftData::SpectrumAVX2 with one fmaf per bin.
func kernSpectrumAvx2Spec(re, im, out *[FftLengthBy2Plus1]float32) {
	for k := 0; k < FftLengthBy2; k++ {
		out[k] = fmaf(re[k], re[k], float32(im[k]*im[k]))
	}
	out[FftLengthBy2] = float32(re[FftLengthBy2]*re[FftLengthBy2]) + float32(im[FftLengthBy2]*im[FftLengthBy2])
}

// TestKernSpectrumAvx2RefMatchesSpec checks kernSpectrumAvx2Ref on plain
// values, rounding ties, results below 2^-126, zeros, Inf and NaN at every
// position of a group, also with out aliasing re or im.
func TestKernSpectrumAvx2RefMatchesSpec(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	tie := float32(4097.0 / 4096.0)
	special := func(v *[FftLengthBy2Plus1]float32, isRe bool) {
		for n := rng.Intn(4); n > 0; n-- {
			k := rng.Intn(FftLengthBy2Plus1)
			switch rng.Intn(6) {
			case 0:
				if isRe {
					v[k] = tie
				} else {
					v[k] = 0
				}
			case 1:
				v[k] = float32(rng.NormFloat64() * 1e-20)
			case 2:
				v[k] = float32(rng.NormFloat64() * 1e-23)
			case 3:
				v[k] = 0
			case 4:
				v[k] = float32(math.Inf(1 - 2*rng.Intn(2)))
			default:
				v[k] = float32(math.NaN())
			}
		}
	}
	slow := 0
	for trial := 0; trial < 20000; trial++ {
		var re, im, got, want [FftLengthBy2Plus1]float32
		copy(re[:], kernRandVec(rng, FftLengthBy2Plus1))
		copy(im[:], kernRandVec(rng, FftLengthBy2Plus1))
		if trial%4 != 0 {
			special(&re, true)
			special(&im, false)
		}
		if trial%16 == 5 {
			// A tie in re with im = 0 at a chosen bin.
			k := rng.Intn(FftLengthBy2)
			re[k], im[k] = tie, 0
		}
		for k := 0; k < FftLengthBy2; k++ {
			if fmafHard(fmafSum(re[k], re[k], float32(im[k]*im[k]))) {
				slow++
				break
			}
		}
		copy(got[:], kernRandVec(rng, FftLengthBy2Plus1))
		kernSpectrumAvx2Spec(&re, &im, &want)
		kernSpectrumAvx2Ref(&re, &im, &got)
		kernCheckExact(t, "SpectrumAvx2Ref", got[:], want[:])
		r2, i2 := re, im
		kernSpectrumAvx2Ref(&r2, &i2, &r2)
		kernCheckExact(t, "SpectrumAvx2Ref in place (re)", r2[:], want[:])
		r2 = re
		kernSpectrumAvx2Ref(&r2, &i2, &i2)
		kernCheckExact(t, "SpectrumAvx2Ref in place (im)", i2[:], want[:])
	}
	if slow < 1000 {
		t.Fatalf("only %d inputs reached the slow path", slow)
	}
}

func TestKernReferencesMatchSpec(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for nr := 0; nr <= 40; nr++ {
		for trial := 0; trial < 20; trial++ {
			rows := make([]*[FftLengthBy2Plus1]float32, nr)
			for i := range rows {
				rows[i] = new([FftLengthBy2Plus1]float32)
				copy(rows[i][:], kernEdgeVec(rng, FftLengthBy2Plus1))
			}
			var got, want [FftLengthBy2Plus1]float32
			copy(got[:], kernRandVec(rng, FftLengthBy2Plus1))
			kernSumRowsGeneric(&got, rows)
			kernSumRowsSpec(&want, rows)
			kernCheckExact(t, "SumRows", got[:], want[:])
		}
	}
	for trial := 0; trial < 2000; trial++ {
		var v [FftLength]float32
		copy(v[:], kernEdgeVec(rng, FftLength))
		var gre, gim, wre, wim [FftLengthBy2Plus1]float32
		copy(gre[:], kernRandVec(rng, FftLengthBy2Plus1))
		copy(gim[:], kernRandVec(rng, FftLengthBy2Plus1))
		wre, wim = gre, gim
		kernUnpackGeneric(&v, &gre, &gim)
		kernUnpackSpec(&v, &wre, &wim)
		kernCheckExact(t, "Unpack re", gre[:], wre[:])
		kernCheckExact(t, "Unpack im", gim[:], wim[:])

		copy(gre[:], kernEdgeVec(rng, FftLengthBy2Plus1))
		copy(gim[:], kernEdgeVec(rng, FftLengthBy2Plus1))
		var gv, wv [FftLength]float32
		copy(gv[:], kernRandVec(rng, FftLength))
		wv = gv
		kernPackGeneric(&gre, &gim, &gv)
		kernPackSpec(&gre, &gim, &wv)
		kernCheckExact(t, "Pack", gv[:], wv[:])
	}
	for trial := 0; trial < 3000; trial++ {
		n := 3 + rng.Intn(200)
		x := kernEdgeVec(rng, n)
		h := [3]float32{float32(rng.NormFloat64()), float32(rng.NormFloat64()), float32(rng.NormFloat64())}
		if trial%7 == 0 {
			h = [3]float32{0.7929742, -0.36072128, -0.47047766}
		}
		from := 2 + rng.Intn(n-2)
		to := from - 1 + rng.Intn(n-from+1) // from-1 (empty) up to n-1
		got := kernRandVec(rng, n)
		want := append([]float32(nil), got...)
		kernFir3Generic(x, &h, got, from, to)
		kernFir3Spec(x, &h, want, from, to)
		kernCheckExact(t, "Fir3", got, want)
	}
}

// kernErlUpdateSpec is the bin loops of ErlEstimator::Update with the hold
// counters laid out as in the C++: hold_counters_[k - 1] for bin k.
func kernErlUpdateSpec(X2, Y2, erl *[FftLengthBy2Plus1]float32, holdCounters *[FftLengthBy2Minus1]int32) {
	const x2Min float32 = 44015068.0
	for k := 1; k < FftLengthBy2; k++ {
		if X2[k] > x2Min {
			newErl := Y2[k] / X2[k]
			if newErl < erl[k] {
				holdCounters[k-1] = 1000
				erl[k] += float32(0.1 * (newErl - erl[k]))
				erl[k] = stdMax(erl[k], stMinErl)
			}
		}
	}
	for i := range holdCounters {
		holdCounters[i]--
	}
	for i, a := range holdCounters {
		if a <= 0 {
			erl[i+1] = stdMin(stMaxErl, 2*erl[i+1])
		}
	}
}

// TestKernErlUpdate checks the kernel and its reference against
// kernErlUpdateSpec: every ERL bin but bin 0, which the caller replaces, and
// every hold counter. The render powers lie around the update threshold, the
// hold counters around the release and the ERLs around their bounds.
func TestKernErlUpdate(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	updates, releases := 0, 0
	for trial := 0; trial < 20000; trial++ {
		var X2, Y2, erl [FftLengthBy2Plus1]float32
		var hold [FftLengthBy2]int32
		var spec [FftLengthBy2Minus1]int32
		for k := range X2 {
			X2[k] = float32(44015068.0 * math.Exp(4*rng.NormFloat64()))
			Y2[k] = X2[k] * float32(math.Exp(3*rng.NormFloat64()))
			erl[k] = float32(math.Exp(3 * rng.NormFloat64()))
			switch rng.Intn(12) {
			case 0:
				X2[k] = 44015068.0
			case 1:
				X2[k] = 0
			case 2:
				erl[k] = stMaxErl
			case 3:
				erl[k] = stMinErl
			case 4:
				Y2[k] = 0
			case 5:
				erl[k] = 600
			}
		}
		if trial%3 == 0 {
			copy(X2[:], kernEdgeVec(rng, FftLengthBy2Plus1))
			copy(Y2[:], kernEdgeVec(rng, FftLengthBy2Plus1))
		}
		for k := 1; k < FftLengthBy2; k++ {
			switch rng.Intn(4) {
			case 0:
				hold[k] = int32(rng.Intn(5)) - 2
			case 1:
				hold[k] = int32(rng.Intn(1001))
			case 2:
				hold[k] = math.MinInt32 + int32(rng.Intn(3))
			}
			spec[k-1] = hold[k]
		}
		gErl, wErl, rErl := erl, erl, erl
		gHold, rHold := hold, hold
		kernErlUpdate(&X2, &Y2, &gErl, &gHold)
		kernErlUpdateGeneric(&X2, &Y2, &rErl, &rHold)
		kernErlUpdateSpec(&X2, &Y2, &wErl, &spec)
		gErl[0], rErl[0] = wErl[0], wErl[0]
		kernCheckExact(t, "ErlUpdate", gErl[:], wErl[:])
		kernCheckExact(t, "ErlUpdateGeneric", rErl[:], wErl[:])
		if gHold[0] != 0 || rHold[0] != 0 {
			t.Fatalf("hold[0] = %d, %d", gHold[0], rHold[0])
		}
		for k := 1; k < FftLengthBy2; k++ {
			if gHold[k] != spec[k-1] || rHold[k] != spec[k-1] {
				t.Fatalf("trial %d hold[%d] = %d (kernel), %d (reference), want %d", trial, k, gHold[k], rHold[k], spec[k-1])
			}
			if spec[k-1] == 999 {
				updates++
			}
			if spec[k-1] <= 0 {
				releases++
			}
		}
	}
	if updates < 10000 || releases < 10000 {
		t.Fatalf("%d updates and %d releases", updates, releases)
	}
}
