package aec3

import (
	"math/rand"
	"testing"
)

// The references below that are not a plain element loop are checked against
// the loop they stand for, in every build (without archsimd the kernels are
// the references themselves, so the kernel tests do not cover them).

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
