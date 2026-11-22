package aec3

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

var (
	mmNaN  = float32(math.NaN())
	mmNegZ = float32(math.Copysign(0, -1))
	mmInf  = float32(math.Inf(1))
)

func mmBits(v float32) uint32 { return math.Float32bits(v) }

// TestStdMinMaxSemantics checks the helpers against the C++ expressions:
// std::min(a, b) = b < a ? b : a, std::max(a, b) = a < b ? b : a,
// _mm_max_ps(a, b) = a > b ? a : b, SafeClamp and fabsf.
func TestStdMinMaxSemantics(t *testing.T) {
	type c2 struct{ a, b, min, max, sse float32 }
	for _, c := range []c2{
		{1, 2, 1, 2, 2},
		{2, 1, 1, 2, 2},
		{1, mmNaN, 1, 1, mmNaN},
		{mmNaN, 1, mmNaN, mmNaN, 1},
		{mmNegZ, 0, mmNegZ, mmNegZ, 0},
		{0, mmNegZ, 0, 0, mmNegZ},
		{-mmInf, mmInf, -mmInf, mmInf, mmInf},
	} {
		check := func(name string, got, want float32) {
			t.Helper()
			if mmBits(got) != mmBits(want) && !(got != got && want != want) {
				t.Errorf("%s(%v, %v) = %v (%#x), want %v (%#x)", name, c.a, c.b, got, mmBits(got), want, mmBits(want))
			}
		}
		check("stdMin", stdMin(c.a, c.b), c.min)
		check("stdMax", stdMax(c.a, c.b), c.max)
		check("sseMax", sseMax(c.a, c.b), c.sse)
	}
	if v := stSafeClamp(mmNegZ, 0, 1); mmBits(v) != 0 {
		t.Errorf("SafeClamp(-0, 0, 1) = %#x, want +0", mmBits(v))
	}
	if v := stSafeClamp(mmNaN, 0, 1); v == v {
		t.Errorf("SafeClamp(NaN, 0, 1) = %v, want NaN", v)
	}
	for _, v := range []float32{mmNegZ, -1, mmNaN, -mmInf} {
		if a := fabsf(v); mmBits(a)>>31 != 0 {
			t.Errorf("fabsf(%v) keeps the sign: %#x", v, mmBits(a))
		}
	}
	x := []float32{mmNegZ, 0, 3, mmNaN, -2}
	kernClampGeneric(x, 0, 1)
	for i, want := range []float32{mmNegZ, 0, 1, mmNaN, 0} {
		if mmBits(x[i]) != mmBits(want) && !(x[i] != x[i] && want != want) {
			t.Errorf("kernClampGeneric idx %d = %v (%#x), want %v", i, x[i], mmBits(x[i]), want)
		}
	}
	// std::max(0.f, NaN) is 0: a NaN tmp in the audibility weighting zeroes
	// the weighted echo.
	e := []float32{1}
	w := []float32{-1}
	kernWeighGeneric(2, mmNaN, e, w)
	if w[0] != 0 {
		t.Errorf("weigh with NaN normalizer = %v, want 0", w[0])
	}
}

// TestStdMinmaxElement checks stdMinmaxElement: the first minimum and the
// last maximum without a NaN, the result of the pairing with one.
func TestStdMinmaxElement(t *testing.T) {
	type cs struct {
		x      []float32
		mn, mx float32
	}
	for _, c := range []cs{
		{[]float32{5}, 5, 5},
		{[]float32{1, mmNaN, 0}, 0, mmNaN},
		{[]float32{mmNaN, 1, 2}, mmNaN, 2},
		{[]float32{1, 2, mmNaN, 0}, 1, 2},
		{[]float32{1, 2, 0, mmNaN}, 0, mmNaN},
		{[]float32{0, mmNegZ, mmNegZ, 0}, 0, 0},
		{[]float32{mmNegZ, 0, 0, mmNegZ, 0}, mmNegZ, 0},
	} {
		mn, mx := stdMinmaxElement(c.x)
		if mmBits(mn) != mmBits(c.mn) && !(mn != mn && c.mn != c.mn) ||
			mmBits(mx) != mmBits(c.mx) && !(mx != mx && c.mx != c.mx) {
			t.Errorf("minmax_element(%v) = (%v, %v), want (%v, %v)", c.x, mn, mx, c.mn, c.mx)
		}
	}
	// Without NaN: the first minimum and the last maximum, signed zeros by
	// position.
	rng := rand.New(rand.NewSource(3))
	vals := []float32{mmNegZ, 0, 1, -1, 2}
	for trial := 0; trial < 20000; trial++ {
		x := make([]float32, 1+rng.Intn(12))
		for i := range x {
			x[i] = vals[rng.Intn(len(vals))]
		}
		wmn, wmx := x[0], x[0]
		for _, v := range x[1:] {
			if v < wmn {
				wmn = v
			}
			if v >= wmx {
				wmx = v
			}
		}
		mn, mx := stdMinmaxElement(x)
		if mmBits(mn) != mmBits(wmn) || mmBits(mx) != mmBits(wmx) {
			t.Fatalf("minmax_element(%v) = (%v, %v), want (%v, %v)", x, mn, mx, wmn, wmx)
		}
		// The fast path agrees whenever it answers.
		if fmn, fmx, ok := kernMinMax(x); ok && (mmBits(fmn) != mmBits(mn) || mmBits(fmx) != mmBits(mx)) {
			t.Fatalf("kernMinMax(%v) = (%v, %v), want (%v, %v)", x, fmn, fmx, mn, mx)
		}
	}
}

// mmEdgeFft fills d with values that include NaN, ±Inf and ±0.
func mmEdgeFft(rng *rand.Rand, d *FftData) {
	copy(d.Re[:], kernEdgeVec(rng, FftLengthBy2Plus1))
	copy(d.Im[:], kernEdgeVec(rng, FftLengthBy2Plus1))
	for k := range d.Re {
		if rng.Intn(16) == 0 {
			d.Re[k], d.Im[k] = 0, mmNegZ
		}
	}
}

// TestAfirFreqRespLaneSemantics checks the frequency response block of every
// lane, on NaN, Inf and signed zeros, against the lane references:
// vmaxq_f32, which propagates a NaN, _mm_max_ps and std::max.
func TestAfirFreqRespLaneSemantics(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	lanes := []struct {
		opt Aec3Optimization
		ref func(*FftData, *[FftLengthBy2Plus1]float32)
	}{
		{OptNone, func(h *FftData, h2 *[FftLengthBy2Plus1]float32) { afirFreqRespBinsGo(h, h2, 0, FftLengthBy2) }},
		{OptNeon, afirFreqRespBlockNeonGo},
		{OptSse2, afirFreqRespBlockSse2Go},
		{OptAvx2, afirFreqRespBlockAvx2Go},
	}
	for trial := 0; trial < 3000; trial++ {
		var h FftData
		mmEdgeFft(rng, &h)
		var h2 [FftLengthBy2Plus1]float32
		copy(h2[:], kernEdgeVec(rng, FftLengthBy2Plus1))
		for _, l := range lanes {
			got, want := h2, h2
			afirFreqRespBlock(l.opt, &h, &got)
			l.ref(&h, &want)
			kernCheckExact(t, fmt.Sprintf("freqresp lane %d", l.opt), got[:], want[:])
		}
		for _, opt := range []Aec3Optimization{OptSse2, OptAvx2} {
			if !afirHasX86FreqResp(opt) {
				continue
			}
			hs := make([]FftData, 1+rng.Intn(3))
			for i := range hs {
				mmEdgeFft(rng, &hs[i])
			}
			var got, want [FftLengthBy2Plus1]float32
			copy(got[:], kernEdgeVec(rng, FftLengthBy2Plus1))
			afirFreqRespPartitionX86(opt, hs, &got)
			for i := range hs {
				if opt == OptSse2 {
					afirFreqRespBlockSse2Go(&hs[i], &want)
				} else {
					afirFreqRespBlockAvx2Go(&hs[i], &want)
				}
				afirFreqRespBinsGo(&hs[i], &want, FftLengthBy2, FftLengthBy2Plus1)
			}
			kernCheckExact(t, fmt.Sprintf("freqresp partition x86 lane %d", opt), got[:], want[:])
		}
		if afirHasNeonFreqResp {
			hs := make([]FftData, 1+rng.Intn(3))
			for i := range hs {
				mmEdgeFft(rng, &hs[i])
			}
			var got, want [FftLengthBy2Plus1]float32
			afirFreqRespPartitionNEON(hs, &got)
			for i := range hs {
				afirFreqRespBlockNeonGo(&hs[i], &want)
				afirFreqRespBinsGo(&hs[i], &want, FftLengthBy2, FftLengthBy2Plus1)
			}
			kernCheckExact(t, "freqresp partition neon", got[:], want[:])
		}
	}
	// A NaN in the stored maximum survives vmaxq_f32 and std::max;
	// _mm_max_ps replaces it.
	var h FftData
	h.Re[3] = 2
	for _, l := range lanes {
		var h2 [FftLengthBy2Plus1]float32
		h2[3] = mmNaN
		l.ref(&h, &h2)
		if nan := h2[3] != h2[3]; nan != (l.opt != OptSse2 && l.opt != OptAvx2) {
			t.Errorf("%v: max(NaN, 4) = %v", l.opt, h2[3])
		}
	}
}
