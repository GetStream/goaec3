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

// The pure-Go implementations of the builds without archsimd kernels are
// bit-identical to the transcriptions of the C sources they replace, in
// every build.
func TestFastMatchesRef(t *testing.T) {
	compareFfts(t, "lane/fwd/neon", fftFastLane, fftNeonRef)
	compareFfts(t, "lane/inv/neon", inverseFftFastLane, inverseFftNeonRef)
	compareFfts(t, "lane/fwd/sse2", fftFastLane, fftSSE2Ref)
	compareFfts(t, "lane/inv/sse2", inverseFftFastLane, inverseFftSSE2Ref)
	compareFfts(t, "scalar/fwd", fftFastScalar, fftScalar)
	compareFfts(t, "scalar/inv", inverseFftFastScalar, inverseFftScalar)
	compareFfts(t, "build/fwd", fftFast, fftRef)
	compareFfts(t, "build/inv", inverseFftFast, inverseFftRef)
}

func TestFftMatchesLaneRef(t *testing.T) {
	compareFfts(t, "fwd", Fft, fftRef)
	compareFfts(t, "inv", InverseFft, inverseFftRef)
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

func TestFftNoAllocs(t *testing.T) {
	var a [128]float32
	if n := testing.AllocsPerRun(100, func() { Fft(&a); InverseFft(&a) }); n != 0 {
		t.Fatalf("Fft allocates %v times", n)
	}
}

func benchFft(b *testing.B, f fftFunc) {
	var a [128]float32
	rng := rand.New(rand.NewSource(1))
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	var x [128]float32
	b.ReportAllocs()
	for b.Loop() {
		x = a
		f(&x)
	}
}

func BenchmarkFft(b *testing.B) {
	b.Run("lane", func(b *testing.B) { benchFft(b, Fft) })
	b.Run("fastLane", func(b *testing.B) { benchFft(b, fftFastLane) })
	b.Run("fastScalar", func(b *testing.B) { benchFft(b, fftFastScalar) })
	b.Run("scalar", func(b *testing.B) { benchFft(b, fftScalar) })
	b.Run("neonRef", func(b *testing.B) { benchFft(b, fftNeonRef) })
	b.Run("sse2Ref", func(b *testing.B) { benchFft(b, fftSSE2Ref) })
}

func BenchmarkInverseFft(b *testing.B) {
	b.Run("lane", func(b *testing.B) { benchFft(b, InverseFft) })
	b.Run("fastLane", func(b *testing.B) { benchFft(b, inverseFftFastLane) })
	b.Run("fastScalar", func(b *testing.B) { benchFft(b, inverseFftFastScalar) })
	b.Run("scalar", func(b *testing.B) { benchFft(b, inverseFftScalar) })
	b.Run("neonRef", func(b *testing.B) { benchFft(b, inverseFftNeonRef) })
	b.Run("sse2Ref", func(b *testing.B) { benchFft(b, inverseFftSSE2Ref) })
}
