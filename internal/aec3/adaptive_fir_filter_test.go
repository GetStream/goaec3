package aec3

import (
	"math"
	"math/rand"
	"testing"
)

func afirTestRandFft(rng *rand.Rand, d *FftData, scale float64) {
	for k := range d.Re {
		d.Re[k] = float32(rng.NormFloat64() * scale)
		d.Im[k] = float32(rng.NormFloat64() * scale)
	}
	d.Im[0], d.Im[FftLengthBy2] = 0, 0
}

func afirTestRenderBuffer(rng *rand.Rand, size, numChannels, read int) *RenderBuffer {
	fb := NewFftBuffer(size, numChannels)
	for i := range fb.Size {
		for ch := range fb.At(i) {
			afirTestRandFft(rng, &fb.At(i)[ch], 100)
		}
	}
	fb.Read = read
	return NewRenderBuffer(NewBlockBuffer(size, 1, numChannels), NewSpectrumBuffer(size, numChannels), fb)
}

func afirTestH(rng *rand.Rand, numPartitions, numChannels int) FirCoefficients {
	H := NewFirCoefficients(numPartitions, numChannels)
	for i := range H.Flat() {
		afirTestRandFft(rng, &H.Flat()[i], 0.1)
	}
	return H
}

func afirEqualFft(t *testing.T, what string, got, want *FftData) {
	t.Helper()
	for k := range got.Re {
		if math.Float32bits(got.Re[k]) != math.Float32bits(want.Re[k]) || math.Float32bits(got.Im[k]) != math.Float32bits(want.Im[k]) {
			t.Fatalf("%s: bin %d got (%v,%v) want (%v,%v)", what, k, got.Re[k], got.Im[k], want.Re[k], want.Im[k])
		}
	}
}

func afirEqualArr(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", what, len(got), len(want))
	}
	for k := range got {
		if math.Float32bits(got[k]) != math.Float32bits(want[k]) {
			t.Fatalf("%s: idx %d got %v want %v", what, k, got[k], want[k])
		}
	}
}

// The NEON/SSE2/AVX2 ApplyFilter and AdaptPartitions lanes (and every
// ErlComputer lane, and the NEON/SSE2 frequency responses) perform the same
// rounded operations per bin as the scalar code; in SIMD builds this also
// proves the archsimd kernels against the scalar reference.
func TestAfirLanesMatchScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	type applyFn func(*RenderBuffer, int, FirCoefficients, *FftData)
	type adaptFn func(*RenderBuffer, *FftData, int, FirCoefficients)
	type frFn func(int, FirCoefficients, [][FftLengthBy2Plus1]float32)
	type erlFn func([][FftLengthBy2Plus1]float32, []float32)
	applies := map[string]applyFn{"neon": ApplyFilterNeon, "sse2": ApplyFilterSse2, "avx2": ApplyFilterAvx2}
	adapts := map[string]adaptFn{"neon": AdaptPartitionsNeon, "sse2": AdaptPartitionsSse2, "avx2": AdaptPartitionsAvx2}
	frs := map[string]frFn{"neon": ComputeFrequencyResponseNeon, "sse2": ComputeFrequencyResponseSse2}
	erls := map[string]erlFn{"neon": ErlComputerNEON, "sse2": ErlComputerSSE2, "avx2": ErlComputerAVX2}
	for trial := 0; trial < 200; trial++ {
		size := 14 + rng.Intn(10)
		numCh := 1 + rng.Intn(3)
		read := rng.Intn(size)
		np := rng.Intn(size + 1)
		rb := afirTestRenderBuffer(rng, size, numCh, read)
		H := afirTestH(rng, size, numCh)
		var G FftData
		afirTestRandFft(rng, &G, 0.01)

		var want FftData
		ApplyFilter(rb, np, H, &want)
		for name, f := range applies {
			var got FftData
			got.Re[3] = 7 // must be cleared
			f(rb, np, H, &got)
			afirEqualFft(t, "ApplyFilter "+name, &got, &want)
		}

		Hw := afirCopyH(H)
		AdaptPartitions(rb, &G, np, Hw)
		for name, f := range adapts {
			Hg := afirCopyH(H)
			f(rb, &G, np, Hg)
			for p := range Hg.NumPartitions() {
				for ch := range Hg.At(p) {
					afirEqualFft(t, "AdaptPartitions "+name, &Hg.At(p)[ch], &Hw.At(p)[ch])
				}
			}
		}

		H2w := make([][FftLengthBy2Plus1]float32, size)
		ComputeFrequencyResponse(np, H, H2w)
		for name, f := range frs {
			H2g := make([][FftLengthBy2Plus1]float32, size)
			H2g[0][5] = 3
			f(np, H, H2g)
			for p := range H2g {
				afirEqualArr(t, "FreqResp "+name, H2g[p][:], H2w[p][:])
			}
		}

		erlw := make([]float32, FftLengthBy2Plus1)
		ErlComputer(H2w, erlw)
		for name, f := range erls {
			erlg := make([]float32, FftLengthBy2Plus1)
			erlg[2] = 1
			f(H2w, erlg)
			afirEqualArr(t, "Erl "+name, erlg, erlw)
		}
	}
}

// The bounds-check-free and term-ordered pure-Go kernels against the plain
// per-bin references.
func TestAfirFastGoKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 300; trial++ {
		n := rng.Intn(40)
		xs := make([]*FftData, n)
		hs := make([]*FftData, n)
		hw := make([]*FftData, n)
		for i := range xs {
			xs[i], hs[i], hw[i] = new(FftData), new(FftData), new(FftData)
			afirTestRandFft(rng, xs[i], 100)
			afirTestRandFft(rng, hs[i], 0.1)
			if trial%5 == 0 {
				// Signed zeros and non-finite values keep their place.
				k := rng.Intn(FftLengthBy2Plus1)
				xs[i].Re[k] = float32(math.Copysign(0, -1))
				hs[i].Im[rng.Intn(FftLengthBy2Plus1)] = float32(math.Inf(1 - 2*rng.Intn(2)))
				hs[i].Re[rng.Intn(FftLengthBy2Plus1)] = float32(math.Copysign(0, -1))
			}
			*hw[i] = *hs[i]
		}
		var G FftData
		afirTestRandFft(rng, &G, 0.01)

		var got, want FftData
		got.Re[9], got.Im[64] = 5, 6 // must be overwritten
		afirApplyTermsGo(xs, hs, &want)
		afirApplyTermsFastGo(xs, hs, &got)
		afirEqualFft(t, "apply terms", &got, &want)
		got.Clear()
		for i := range xs {
			afirApplyAllGo(xs[i], hs[i], &got)
		}
		afirEqualFft(t, "apply all", &got, &want)

		afirAdaptTermsGo(xs, hw, &G)
		afirAdaptTermsFastGo(xs, hs, &G)
		for i := range hs {
			afirEqualFft(t, "adapt terms", hs[i], hw[i])
		}
	}
}

func TestAfirMaxAbsUpdate(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	for trial := 0; trial < 500; trial++ {
		var ir, h, want [FftLengthBy2]float32
		copy(ir[:], kernEdgeVec(rng, FftLengthBy2))
		copy(h[:], kernEdgeVec(rng, FftLengthBy2))
		for k := range ir {
			switch rng.Intn(8) {
			case 0:
				h[k] = ir[k]
			case 1:
				h[k] = -ir[k]
			}
		}
		want = ir
		afirMaxAbsUpdateGo(&want, &h)
		afirMaxAbsUpdate(&ir, &h)
		for k := range ir {
			if math.Float32bits(ir[k]) != math.Float32bits(want[k]) {
				t.Fatalf("trial %d idx %d: got %v (%#x) want %v (%#x)", trial, k, ir[k], math.Float32bits(ir[k]),
					want[k], math.Float32bits(want[k]))
			}
		}
	}
}

func BenchmarkAfirMaxAbsUpdate(b *testing.B) {
	rng := rand.New(rand.NewSource(13))
	var ir, h [FftLengthBy2]float32
	for k := range ir {
		ir[k], h[k] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	b.Run("dispatch", func(b *testing.B) {
		for b.Loop() {
			afirMaxAbsUpdate(&ir, &h)
		}
	})
	b.Run("ref", func(b *testing.B) {
		for b.Loop() {
			afirMaxAbsUpdateGo(&ir, &h)
		}
	})
}

func afirCopyH(H FirCoefficients) FirCoefficients { return H.Clone() }

// The AVX2 frequency response (fused) against its pure-Go reference; in
// amd64 SIMD builds with AVX2+FMA this checks the archsimd kernel.
func TestAfirFreqRespAvx2(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	im := float32(4097.0 / 4096.0)
	for trial := 0; trial < 300; trial++ {
		np := 1 + rng.Intn(13)
		numCh := 1 + rng.Intn(2)
		H := afirTestH(rng, np, numCh)
		if trial%3 == 0 {
			for j := 0; j < FftLengthBy2; j++ {
				H.At(0)[0].Im[j] = im
				H.At(0)[0].Re[j] = float32(1e-17 * float64(j+1))
			}
		}
		want := make([][FftLengthBy2Plus1]float32, np)
		for p := 0; p < np; p++ {
			for ch := 0; ch < numCh; ch++ {
				afirFreqRespBlockAvx2Go(&H.At(p)[ch], &want[p])
				afirFreqRespBinsGo(&H.At(p)[ch], &want[p], FftLengthBy2, FftLengthBy2Plus1)
			}
		}
		got := make([][FftLengthBy2Plus1]float32, np)
		ComputeFrequencyResponseAvx2(np, H, got)
		for p := range got {
			afirEqualArr(t, "avx2 freqresp", got[p][:], want[p][:])
		}
	}
}

func TestAfirAdaptiveFirFilterNoAllocs(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, opt := range []Aec3Optimization{OptNone, DetectOptimization()} {
		f := NewAdaptiveFirFilter(13, 12, 250, 2, opt)
		rb := afirTestRenderBuffer(rng, 20, 2, 15)
		var G, S FftData
		afirTestRandFft(rng, &G, 0.001)
		ir := make([]float32, 0, GetTimeDomainLength(13))
		H2 := make([][FftLengthBy2Plus1]float32, 0, 13)
		erl := make([]float32, FftLengthBy2Plus1)
		f.SetSizePartitions(13, false)
		allocs := testing.AllocsPerRun(100, func() {
			f.Filter(rb, &S)
			f.AdaptWithImpulseResponse(rb, &G, &ir)
			f.Adapt(rb, &G)
			f.ComputeFrequencyResponse(&H2)
			ComputeErl(opt, H2, erl)
		})
		if allocs != 0 {
			t.Fatalf("opt %d: %v allocs", opt, allocs)
		}
	}
}

func afirBenchSetup(b *testing.B) (*RenderBuffer, FirCoefficients, *FftData) {
	rng := rand.New(rand.NewSource(6))
	rb := afirTestRenderBuffer(rng, 20, 1, 15)
	H := afirTestH(rng, 13, 1)
	var G FftData
	afirTestRandFft(rng, &G, 0.001)
	return rb, H, &G
}

var afirBenchLanes = []struct {
	name string
	opt  Aec3Optimization
}{{"None", OptNone}, {"Neon", OptNeon}, {"Sse2", OptSse2}, {"Avx2", OptAvx2}}

func BenchmarkAfirApplyFilter(b *testing.B) {
	rb, H, _ := afirBenchSetup(b)
	for _, l := range afirBenchLanes {
		b.Run(l.name, func(b *testing.B) {
			f := NewAdaptiveFirFilter(13, 13, 250, 1, l.opt)
			f.SetFilter(13, H)
			var S FftData
			for b.Loop() {
				f.Filter(rb, &S)
			}
		})
	}
}

func BenchmarkAfirAdaptPartitions(b *testing.B) {
	rb, H, G := afirBenchSetup(b)
	fns := map[string]func(*RenderBuffer, *FftData, int, FirCoefficients){
		"None": AdaptPartitions, "Neon": AdaptPartitionsNeon, "Sse2": AdaptPartitionsSse2, "Avx2": AdaptPartitionsAvx2,
	}
	for _, l := range afirBenchLanes {
		b.Run(l.name, func(b *testing.B) {
			for b.Loop() {
				fns[l.name](rb, G, 13, H)
			}
		})
	}
}

func BenchmarkAfirComputeFrequencyResponse(b *testing.B) {
	_, H, _ := afirBenchSetup(b)
	H2 := make([][FftLengthBy2Plus1]float32, 13)
	fns := map[string]func(int, FirCoefficients, [][FftLengthBy2Plus1]float32){
		"None": ComputeFrequencyResponse, "Neon": ComputeFrequencyResponseNeon, "Sse2": ComputeFrequencyResponseSse2, "Avx2": ComputeFrequencyResponseAvx2,
	}
	for _, l := range afirBenchLanes {
		b.Run(l.name, func(b *testing.B) {
			for b.Loop() {
				fns[l.name](13, H, H2)
			}
		})
	}
}

func BenchmarkAfirErlComputer(b *testing.B) {
	_, H, _ := afirBenchSetup(b)
	H2 := make([][FftLengthBy2Plus1]float32, 13)
	ComputeFrequencyResponse(13, H, H2)
	erl := make([]float32, FftLengthBy2Plus1)
	for _, l := range afirBenchLanes {
		b.Run(l.name, func(b *testing.B) {
			for b.Loop() {
				ComputeErl(l.opt, H2, erl)
			}
		})
	}
}
