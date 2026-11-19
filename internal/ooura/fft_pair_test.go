package ooura

import "testing"

type fftPairFunc func(a, b *[128]float32, s *PairScratch)

// comparePairs runs pair on every input together with a different input and
// checks both frames against the single transform. The scratch starts
// poisoned with NaNs and is reused, so a result that depends on its contents
// fails.
func comparePairs(t *testing.T, name string, pair fftPairFunc, single fftFunc) {
	t.Helper()
	in := testInputs()
	var s PairScratch
	poisonScratch(&s)
	for n := range in {
		a, b := in[n], in[(n*7+3)%len(in)]
		wa, wb := a, b
		single(&wa)
		single(&wb)
		pair(&a, &b, &s)
		checkSame(t, name+"/a", n, &a, &wa)
		checkSame(t, name+"/b", n, &b, &wb)
	}
}

func TestFftPairMatchesSingle(t *testing.T) {
	comparePairs(t, "fwd", FftPair, Fft)
	comparePairs(t, "inv", InverseFftPair, InverseFft)
	comparePairs(t, "fwd/ref", FftPair, fftRef)
	comparePairs(t, "inv/ref", InverseFftPair, inverseFftRef)
}

func TestFftPairNoAllocs(t *testing.T) {
	var a, b [128]float32
	var s PairScratch
	if n := testing.AllocsPerRun(100, func() { FftPair(&a, &b, &s); InverseFftPair(&a, &b, &s) }); n != 0 {
		t.Fatalf("FftPair allocates %v times", n)
	}
}

func benchFftPair(b *testing.B, f fftPairFunc) {
	in := testInputs()
	x0, y0 := in[0], in[1]
	var x, y [128]float32
	var s PairScratch
	b.ReportAllocs()
	for b.Loop() {
		x, y = x0, y0
		f(&x, &y, &s)
	}
}

func twice(f fftFunc) fftPairFunc {
	return func(a, b *[128]float32, _ *PairScratch) { f(a); f(b) }
}

// BenchmarkFftPair times two transforms: the pair entry points against two
// single calls.
func BenchmarkFftPair(b *testing.B) {
	b.Run("fwd/pair", func(b *testing.B) { benchFftPair(b, FftPair) })
	b.Run("fwd/two", func(b *testing.B) { benchFftPair(b, twice(Fft)) })
	b.Run("inv/pair", func(b *testing.B) { benchFftPair(b, InverseFftPair) })
	b.Run("inv/two", func(b *testing.B) { benchFftPair(b, twice(InverseFft)) })
}
