package aec3

import (
	"math/rand"
	"testing"
)

func fftPairTestSignal(rng *rand.Rand, v []float32) {
	for i := range v {
		v[i] = float32(rng.NormFloat64() * 3000)
	}
}

// The pair transforms are bit-identical to two calls of Aec3Fft.
func TestAec3FftPairMatchesSingle(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	var single Aec3Fft
	var pair aec3FftPair
	for trial := 0; trial < 200; trial++ {
		var x0, x1 [FftLength]float32
		fftPairTestSignal(rng, x0[:])
		fftPairTestSignal(rng, x1[:])
		if trial%5 == 0 {
			clear(x0[:FftLengthBy2])
		}
		var X0, X1, W0, W1 FftData
		a0, a1 := x0, x1
		single.Fft(&a0, &W0)
		single.Fft(&a1, &W1)
		a0, a1 = x0, x1
		pair.Fft(&a0, &a1, &X0, &X1)
		afirEqualFft(t, "Fft/0", &X0, &W0)
		afirEqualFft(t, "Fft/1", &X1, &W1)

		var y0, y1, w0, w1 [FftLength]float32
		single.Ifft(&W0, &w0)
		single.Ifft(&W1, &w1)
		pair.Ifft(&X0, &X1, &y0, &y1)
		afirEqualArr(t, "Ifft/0", y0[:], w0[:])
		afirEqualArr(t, "Ifft/1", y1[:], w1[:])

		single.ZeroPaddedFft(x0[:FftLengthBy2], WindowHanning, &W0)
		single.ZeroPaddedFft(x1[:FftLengthBy2], WindowHanning, &W1)
		pair.ZeroPaddedFft(x0[:FftLengthBy2], x1[:FftLengthBy2], &X0, &X1)
		afirEqualFft(t, "ZeroPaddedFft/0", &X0, &W0)
		afirEqualFft(t, "ZeroPaddedFft/1", &X1, &W1)

		single.PaddedFft(x0[FftLengthBy2:], x0[:FftLengthBy2], WindowSqrtHanning, &W0)
		single.PaddedFft(x1[FftLengthBy2:], x1[:FftLengthBy2], WindowSqrtHanning, &W1)
		pair.PaddedFft(x0[FftLengthBy2:], x0[:FftLengthBy2], x1[FftLengthBy2:], x1[:FftLengthBy2], &X0, &X1)
		afirEqualFft(t, "PaddedFft/0", &X0, &W0)
		afirEqualFft(t, "PaddedFft/1", &X1, &W1)
	}
}

// subPredictionErrorPair is two subPredictionError calls.
func TestSubPredictionErrorPair(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	var single Aec3Fft
	var pair aec3FftPair
	for trial := 0; trial < 100; trial++ {
		var S0, S1 FftData
		afirTestRandFft(rng, &S0, 1000)
		afirTestRandFft(rng, &S1, 1000)
		var y [BlockSize]float32
		fftPairTestSignal(rng, y[:])
		var tmp, tmp2 [FftLength]float32
		var we0, ws0, we1, ws1, e0, s0, e1, s1 [BlockSize]float32
		subPredictionError(&single, &S0, y[:], &tmp, &we0, &ws0)
		subPredictionError(&single, &S1, y[:], &tmp, &we1, &ws1)
		subPredictionErrorPair(&pair, &S0, &S1, y[:], &tmp, &tmp2, &e0, &s0, &e1, &s1)
		afirEqualArr(t, "e0", e0[:], we0[:])
		afirEqualArr(t, "s0", s0[:], ws0[:])
		afirEqualArr(t, "e1", e1[:], we1[:])
		afirEqualArr(t, "s1", s1[:], ws1[:])
	}
}

// BenchmarkAec3Fft times two transforms with their FftData conversion: two
// single calls against one pair call.
func BenchmarkAec3Fft(b *testing.B) {
	rng := rand.New(rand.NewSource(14))
	var x0, x1 [FftLength]float32
	fftPairTestSignal(rng, x0[:])
	fftPairTestSignal(rng, x1[:])
	var single Aec3Fft
	var pair aec3FftPair
	var X0, X1 FftData
	var a0, a1 [FftLength]float32
	b.Run("Fft/two", func(b *testing.B) {
		for b.Loop() {
			a0, a1 = x0, x1
			single.Fft(&a0, &X0)
			single.Fft(&a1, &X1)
		}
	})
	b.Run("Fft/pair", func(b *testing.B) {
		for b.Loop() {
			a0, a1 = x0, x1
			pair.Fft(&a0, &a1, &X0, &X1)
		}
	})
	b.Run("Ifft/two", func(b *testing.B) {
		for b.Loop() {
			single.Ifft(&X0, &a0)
			single.Ifft(&X1, &a1)
		}
	})
	b.Run("Ifft/pair", func(b *testing.B) {
		for b.Loop() {
			pair.Ifft(&X0, &X1, &a0, &a1)
		}
	})
}

// afirConstrainPair leaves both filters, their constraint positions and the
// impulse response exactly as the refined filter's
// constrainAndUpdateImpulseResponse followed by the coarse filter's
// constrain do, over several rounds and filter sizes.
func TestAfirConstrainPair(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	var pair aec3FftPair
	for _, numCh := range []int{1, 2, 3} {
		for _, sizes := range [][2]int{{13, 13}, {12, 5}, {3, 7}} {
			mk := func(n int, H FirCoefficients) *AdaptiveFirFilter {
				f := NewAdaptiveFirFilter(n, n, 250, numCh, DetectOptimization())
				f.SetFilter(n, H)
				return f
			}
			HR := afirTestH(rng, sizes[0], numCh)
			HC := afirTestH(rng, sizes[1], numCh)
			wantR, wantC := mk(sizes[0], HR), mk(sizes[1], HC)
			gotR, gotC := mk(sizes[0], HR), mk(sizes[1], HC)
			wantIr := make([]float32, 0, GetTimeDomainLength(sizes[0]))
			gotIr := make([]float32, 0, GetTimeDomainLength(sizes[0]))
			for round := 0; round < 3*sizes[0]+2; round++ {
				wantR.constrainAndUpdateImpulseResponse(&wantIr)
				wantC.constrain()
				afirConstrainPair(&pair, gotR, &gotIr, gotC)
				if gotR.partitionToConstrain != wantR.partitionToConstrain || gotC.partitionToConstrain != wantC.partitionToConstrain {
					t.Fatalf("ch %d sizes %v round %d: partitions (%d,%d) want (%d,%d)", numCh, sizes, round,
						gotR.partitionToConstrain, gotC.partitionToConstrain, wantR.partitionToConstrain, wantC.partitionToConstrain)
				}
				afirEqualArr(t, "impulse response", gotIr, wantIr)
				for i := range wantR.H.Flat() {
					afirEqualFft(t, "refined H", &gotR.H.Flat()[i], &wantR.H.Flat()[i])
				}
				for i := range wantC.H.Flat() {
					afirEqualFft(t, "coarse H", &gotC.H.Flat()[i], &wantC.H.Flat()[i])
				}
			}
		}
	}
}
