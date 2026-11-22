package ooura

import (
	"math"
	"math/rand"
	"testing"
)

// testInputs returns random frames plus frames with signed zeros,
// zero-padded halves, impulses, small integers and subnormals.
func testInputs() [][128]float32 {
	rng := rand.New(rand.NewSource(7))
	negZero := math.Float32frombits(0x80000000)
	var out [][128]float32
	for trial := 0; trial < 1000; trial++ {
		var a [128]float32
		for i := range a {
			a[i] = float32(rng.NormFloat64() * 3000)
		}
		out = append(out, a)
	}
	var z, nz [128]float32
	for i := range nz {
		nz[i] = negZero
	}
	out = append(out, z, nz)
	for trial := 0; trial < 100; trial++ {
		var lo, hi, quant, sub [128]float32
		for i := 0; i < 64; i++ {
			lo[i] = float32(rng.NormFloat64())
			hi[64+i] = float32(rng.NormFloat64())
		}
		for i := range quant {
			quant[i] = float32(rng.Intn(5) - 2)
			if rng.Intn(3) == 0 {
				quant[i] = negZero
			}
			sub[i] = math.Float32frombits(uint32(rng.Intn(1<<23)) | uint32(rng.Intn(2))<<31)
		}
		out = append(out, lo, hi, quant, sub)
	}
	for i := 0; i < 128; i++ {
		var imp [128]float32
		imp[i] = 1
		out = append(out, imp)
	}
	return out
}

// nonFiniteInputs returns frames with NaNs (of both signs and with
// payloads) and infinities among finite samples.
func nonFiniteInputs() [][128]float32 {
	rng := rand.New(rand.NewSource(9))
	special := []uint32{0x7fc00000, 0xffc00000, 0x7fc12345, 0x7fa00001, 0x7f800000, 0xff800000}
	var out [][128]float32
	for trial := 0; trial < 300; trial++ {
		var a [128]float32
		for i := range a {
			a[i] = float32(rng.NormFloat64() * 3000)
		}
		for n := 1 + rng.Intn(4); n > 0; n-- {
			a[rng.Intn(128)] = math.Float32frombits(special[rng.Intn(len(special))])
		}
		out = append(out, a)
	}
	return out
}

// same32 compares bit-exactly, except that any two NaNs are equal: the sign
// and payload of a NaN that an arithmetic chain produces depend on operand
// order, which compilers choose freely for commutative operations.
func same32(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

func checkSameNaN(t *testing.T, name string, n int, got, want *[128]float32) {
	t.Helper()
	for i := range got {
		if !same32(got[i], want[i]) {
			t.Fatalf("%s input %d idx %d: got %v (%#x) want %v (%#x)", name, n, i,
				got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func checkSame(t *testing.T, name string, n int, got, want *[128]float32) {
	t.Helper()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s input %d idx %d: got %v (%#x) want %v (%#x)", name, n, i,
				got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

type fftFunc func(*[128]float32)

func compareFfts(t *testing.T, name string, f, g fftFunc) {
	t.Helper()
	for n, a := range testInputs() {
		x, y := a, a
		f(&x)
		g(&y)
		checkSame(t, name, n, &x, &y)
	}
}

// The NEON and SSE2 sources compute the same IEEE operations per output
// element, so their transcriptions must agree bit for bit.
func TestSSE2RefMatchesNeonRef(t *testing.T) {
	compareFfts(t, "fwd", fftSSE2Ref, fftNeonRef)
	compareFfts(t, "inv", inverseFftSSE2Ref, inverseFftNeonRef)
}

// The SIMD lanes really differ from the scalar C path (in cft1st/cftmdl's
// first blocks), so a lane mix-up cannot pass the parity tests.
func TestSimdRefDiffersFromScalar(t *testing.T) {
	diff := 0
	for _, a := range testInputs()[:100] {
		x, y := a, a
		fftNeonRef(&x)
		fftScalar(&y)
		if x != y {
			diff++
		}
	}
	if diff == 0 {
		t.Fatal("NEON-order reference never differs from the scalar path")
	}
}

type (
	fftSplitFunc        func(a *[128]float32, re, im *[65]float32)
	inverseFftSplitFunc func(re, im *[65]float32, a *[128]float32)
)

// compareSplit checks the split transforms against the transform followed
// (preceded) by the FftData copy. The outputs start as NaN, and the inverse
// gets NaN in the two elements that the copy does not read.
func compareSplit(t *testing.T, name string, fwd fftSplitFunc, inv inverseFftSplitFunc, f, g fftFunc) {
	t.Helper()
	nan := float32(math.NaN())
	for n, a := range testInputs() {
		var re, im, wre, wim [65]float32
		for i := range re {
			re[i], im[i] = nan, nan
		}
		x, y := a, a
		f(&y)
		unpack(&y, &wre, &wim)
		fwd(&x, &re, &im)
		for i := range re {
			if math.Float32bits(re[i]) != math.Float32bits(wre[i]) || math.Float32bits(im[i]) != math.Float32bits(wim[i]) {
				t.Fatalf("%s fwd input %d bin %d: got (%v, %v) want (%v, %v)", name, n, i, re[i], im[i], wre[i], wim[i])
			}
		}

		// The inverse of the input taken as bins.
		unpack(&a, &re, &im)
		re[64] = a[1]
		im[0], im[64] = nan, nan
		var want [128]float32
		pack(&re, &im, &want)
		g(&want)
		var got [128]float32
		for i := range got {
			got[i] = nan
		}
		inv(&re, &im, &got)
		checkSame(t, name+" inv", n, &got, &want)
	}
}

// splitScratch is the work area of the single split transforms under test.
// It is reused across inputs and poisoned by the tests that have access to
// its contents.
var splitScratch Scratch

func buildFftSplit(a *[128]float32, re, im *[65]float32) { FftSplit(a, re, im, &splitScratch) }

func buildInverseFftSplit(re, im *[65]float32, a *[128]float32) {
	InverseFftSplit(re, im, a, &splitScratch)
}

func TestFftSplitMatchesRef(t *testing.T) {
	compareSplit(t, "build", buildFftSplit, buildInverseFftSplit, fftRef, inverseFftRef)
	compareSplit(t, "fast", fftSplitFast, inverseFftSplitFast, fftRef, inverseFftRef)
}

// The split transforms of the builds without archsimd kernels, which run the
// last passes of the forward transform and the first passes of the inverse
// as one pass, against the transcriptions with the FftData copy: both lanes
// in every build, also on frames with NaN and infinities.
func TestSplitFastMatchesRef(t *testing.T) {
	compareSplit(t, "lane/neon", fftSplitFastLane, inverseFftSplitFastLane, fftNeonRef, inverseFftNeonRef)
	compareSplit(t, "lane/sse2", fftSplitFastLane, inverseFftSplitFastLane, fftSSE2Ref, inverseFftSSE2Ref)
	compareSplit(t, "scalar", fftSplitFastScalar, inverseFftSplitFastScalar, fftScalar, inverseFftScalar)

	lanes := []struct {
		name string
		fwd  fftSplitFunc
		inv  inverseFftSplitFunc
		f, g fftFunc
	}{
		{"lane", fftSplitFastLane, inverseFftSplitFastLane, fftNeonRef, inverseFftNeonRef},
		{"scalar", fftSplitFastScalar, inverseFftSplitFastScalar, fftScalar, inverseFftScalar},
	}
	for _, l := range lanes {
		for n, a := range nonFiniteInputs() {
			var re, im, wre, wim [65]float32
			x, y := a, a
			l.f(&y)
			unpack(&y, &wre, &wim)
			l.fwd(&x, &re, &im)
			for i := range re {
				if !same32(re[i], wre[i]) || !same32(im[i], wim[i]) {
					t.Fatalf("%s fwd non-finite input %d bin %d: got (%v, %v) want (%v, %v)", l.name, n, i, re[i], im[i], wre[i], wim[i])
				}
			}
			unpack(&a, &re, &im)
			re[64] = a[1]
			var got, want [128]float32
			pack(&re, &im, &want)
			l.g(&want)
			l.inv(&re, &im, &got)
			checkSameNaN(t, l.name+" inv non-finite", n, &got, &want)
		}
	}
}

func TestFftRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	var a, orig [128]float32
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	orig = a
	Fft(&a)
	InverseFft(&a)
	for i := range a {
		if d := math.Abs(float64(a[i]*(2.0/128) - orig[i])); d > 1e-5 {
			t.Fatalf("idx %d: %v vs %v", i, a[i]*(2.0/128), orig[i])
		}
	}
}
