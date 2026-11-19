package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// The specs below are the loops of the C++, as the port had them before the
// kernels and the side-by-side folds.

func sumOfSquaresSpec(x []float32) float32 {
	var acc float32
	for _, v := range x {
		acc += float32(v * v)
	}
	return acc
}

func kernEdge64(rng *rand.Rand) *[BlockSize]float32 {
	return (*[BlockSize]float32)(kernEdgeVec(rng, BlockSize))
}

func TestKernSumSq(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	for trial := 0; trial < 2000; trial++ {
		var v [5]*[BlockSize]float32
		for i := range v {
			if trial%2 == 0 {
				v[i] = (*[BlockSize]float32)(kernRandVec(rng, BlockSize))
			} else {
				v[i] = kernEdge64(rng)
			}
		}
		var want [5]float32
		for i := range v {
			want[i] = sumOfSquaresSpec(v[i][:])
		}
		a, b, c, d, e := kernSumSq5(v[0], v[1], v[2], v[3], v[4])
		kernCheckExact(t, "SumSq5", []float32{a, b, c, d, e}, want[:])
		a, b, c = kernSumSq3(v[0], v[1], v[2])
		kernCheckExact(t, "SumSq3", []float32{a, b, c}, want[:3])
	}
}

func TestKernSumMaxSq(t *testing.T) {
	rng := rand.New(rand.NewSource(32))
	negZero := float32(math.Copysign(0, -1))
	starts := []float32{0, negZero, 1, -1, float32(math.Inf(1)), float32(math.NaN()), 1e30}
	for trial := 0; trial < 4000; trial++ {
		x := kernEdge64(rng)
		switch trial % 4 {
		case 1:
			x = (*[BlockSize]float32)(kernRandVec(rng, BlockSize))
		case 2:
			// Ties and zeros of both signs.
			for i := range x {
				x[i] = []float32{0, negZero, 3, -3}[rng.Intn(4)]
			}
		}
		sum := starts[rng.Intn(len(starts))]
		mx := starts[rng.Intn(len(starts))]
		gs, gm := kernSumMaxSq(x, sum, mx)
		ws, wm := kernSumMaxSqGeneric(x, sum, mx)
		kernCheckExact(t, "SumMaxSq", []float32{gs, gm}, []float32{ws, wm})
	}
}

func TestKernStdMinScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(33))
	for n := 0; n <= 70; n++ {
		for trial := 0; trial < 40; trial++ {
			x := kernEdgeVec(rng, n)
			a := kernEdgeVec(rng, 1)[0]
			got, want := append([]float32(nil), x...), append([]float32(nil), x...)
			kernStdMinScalar(got, a)
			kernStdMinScalarGeneric(want, a)
			kernCheckExact(t, "StdMinScalar", got, want)
		}
	}
}

func TestKernNoiseFloorUpdate(t *testing.T) {
	rng := rand.New(rand.NewSource(34))
	for trial := 0; trial < 4000; trial++ {
		hold := int32(rng.Intn(6))
		if trial%50 == 0 {
			hold = []int32{math.MaxInt32, math.MinInt32, -1}[rng.Intn(3)]
		}
		minPower := []float32{0, 1, 1638400, float32(math.NaN())}[rng.Intn(4)]
		p, n := kernEdge65(rng), kernEdge65(rng)
		if trial%2 == 0 {
			p, n = kernNorm65(rng, 1e6), kernNorm65(rng, 1e6)
			for k := range p {
				p[k], n[k] = p[k]*p[k], n[k]*n[k]
				if rng.Intn(8) == 0 {
					p[k] = n[k]
				}
			}
		}
		var c [FftLengthBy2Plus1]int32
		for k := range c {
			c[k] = int32(rng.Intn(8))
			if hold > 8 && rng.Intn(2) == 0 {
				c[k] = hold - int32(rng.Intn(2))
			}
		}
		gn, wn, gc, wc := *n, *n, c, c
		kernNoiseFloorUpdate(p, &gn, &gc, hold, minPower)
		kernNoiseFloorUpdateGeneric(p, &wn, &wc, hold, minPower)
		kernCheckExact(t, "NoiseFloorUpdate", gn[:], wn[:])
		if gc != wc {
			t.Fatalf("NoiseFloorUpdate counters: got %v want %v", gc, wc)
		}
	}
}

// sgBandEnergiesSpec is the energy computation of
// SuppressionGain::UpperBandsGain.
func sgBandEnergiesSpec(render *Block) (lowBandEnergy, highBandEnergy float32) {
	for ch := 0; ch < render.NumChannels(); ch++ {
		lowBandEnergy = stdMax(lowBandEnergy, sumOfSquaresSpec(render.View(0, ch)))
	}
	for k := 1; k < render.NumBands(); k++ {
		for ch := 0; ch < render.NumChannels(); ch++ {
			highBandEnergy = stdMax(highBandEnergy, sumOfSquaresSpec(render.View(k, ch)))
		}
	}
	return lowBandEnergy, highBandEnergy
}

func TestSgBandEnergies(t *testing.T) {
	rng := rand.New(rand.NewSource(35))
	for trial := 0; trial < 3000; trial++ {
		numBands, numChannels := 2+rng.Intn(2), 1+rng.Intn(4)
		b := NewBlock(numBands, numChannels)
		for band := 0; band < numBands; band++ {
			for ch := 0; ch < numChannels; ch++ {
				v := b.View(band, ch)
				switch rng.Intn(4) {
				case 0:
					copy(v, kernEdgeVec(rng, BlockSize))
				case 1:
					// Zero energy, with zeros of both signs.
					for i := range v {
						v[i] = float32(math.Copysign(0, float64(rng.Intn(2))-0.5))
					}
				default:
					copy(v, kernRandVec(rng, BlockSize))
				}
			}
		}
		gl, gh := sgBandEnergies(&b)
		wl, wh := sgBandEnergiesSpec(&b)
		kernCheckExact(t, "sgBandEnergies", []float32{gl, gh}, []float32{wl, wh})
	}
}

// supGenerateComfortNoiseSpec is GenerateComfortNoise of
// comfort_noise_generator.cc after the square root and the level of the
// upper bands.
func supGenerateComfortNoiseSpec(N *[FftLengthBy2Plus1]float32, highBandNoiseLevel float32, seed *uint32, nLow, nHigh *FftData) {
	nLow.Re[0], nLow.Re[FftLengthBy2] = 0, 0
	nHigh.Re[0], nHigh.Re[FftLengthBy2] = 0, 0
	s := *seed
	for k := 1; k < FftLengthBy2; k++ {
		const indexMask = 32 - 1
		// Generate a random 31-bit integer.
		s = (s*69069 + 1) & (0x80000000 - 1)
		// Convert to a 5-bit index.
		i := s >> 26
		// y = sqrt(2) * sin(a)
		x := supSqrt2Sin[i]
		// x = sqrt(2) * cos(a) = sqrt(2) * sin(a + pi/2)
		y := supSqrt2Sin[(i+8)&indexMask]
		// Form low-frequency noise via spectral shaping.
		nLow.Re[k] = N[k] * x
		nLow.Im[k] = N[k] * y
		// Form the high-frequency noise via simple levelling.
		nHigh.Re[k] = highBandNoiseLevel * x
		nHigh.Im[k] = highBandNoiseLevel * y
	}
	*seed = s
}

func TestSupGenerateComfortNoiseMatchesSpec(t *testing.T) {
	rng := rand.New(rand.NewSource(36))
	opt := DetectOptimization()
	for trial := 0; trial < 3000; trial++ {
		N2 := kernNorm65(rng, 1e5)
		for k := range N2 {
			N2[k] *= N2[k]
		}
		if trial%3 == 0 {
			N2 = kernEdge65(rng)
		}
		seed := rng.Uint32()
		if trial%7 == 0 {
			seed = []uint32{0, 42, 0x7fffffff, 0x80000000, 0xffffffff}[rng.Intn(5)]
		}
		var gotLow, gotHigh, wantLow, wantHigh FftData
		for _, d := range []*FftData{&gotLow, &gotHigh} {
			copy(d.Re[:], kernEdgeVec(rng, FftLengthBy2Plus1))
			copy(d.Im[:], kernEdgeVec(rng, FftLengthBy2Plus1))
		}
		wantLow, wantHigh = gotLow, gotHigh

		N := *N2
		NewVectorMath(opt).Sqrt(N[:])
		var sum float32
		for _, v := range N[FftLengthBy2Plus1/2:] {
			sum += v
		}
		wantSeed := seed
		supGenerateComfortNoiseSpec(&N, sum*(1.0/(FftLengthBy2Plus1/2+1)), &wantSeed, &wantLow, &wantHigh)

		gotSeed := seed
		supGenerateComfortNoise(opt, N2, &gotSeed, &gotLow, &gotHigh)
		if gotSeed != wantSeed {
			t.Fatalf("trial %d: seed %#x, want %#x", trial, gotSeed, wantSeed)
		}
		kernCheckExact(t, "low.Re", gotLow.Re[:], wantLow.Re[:])
		kernCheckExact(t, "low.Im", gotLow.Im[:], wantLow.Im[:])
		kernCheckExact(t, "high.Re", gotHigh.Re[:], wantHigh.Re[:])
		kernCheckExact(t, "high.Im", gotHigh.Im[:], wantHigh.Im[:])
	}
}

func TestKernTailNoAllocs(t *testing.T) {
	rng := rand.New(rand.NewSource(37))
	p, n := kernNorm65(rng, 1), kernNorm65(rng, 1)
	var c [FftLengthBy2Plus1]int32
	x := kernEdge64(rng)
	var seed uint32 = 42
	var low, high FftData
	opt := DetectOptimization()
	var sink float32
	if a := testing.AllocsPerRun(100, func() {
		kernNoiseFloorUpdate(p, n, &c, 3, 1)
		kernStdMinScalar(n[17:], 0.5)
		s, m := kernSumMaxSq(x, 0, 0)
		a, b, cc, d, e := kernSumSq5(x, x, x, x, x)
		sink += s + m + a + b + cc + d + e
		supGenerateComfortNoise(opt, p, &seed, &low, &high)
	}); a != 0 {
		t.Fatalf("allocations: %v", a)
	}
}
