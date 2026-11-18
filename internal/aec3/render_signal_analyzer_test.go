package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// refRenderSignalAnalyzer is the branching formulation of the analysis, with
// std::max as stdMax.
type refRenderSignalAnalyzer struct {
	strongPeakFreezeDuration int
	narrowBandCounters       [FftLengthBy2 - 1]int
	narrowPeakBand           int
	hasNarrowPeakBand        bool
	narrowPeakCounter        int
}

func (r *refRenderSignalAnalyzer) Update(renderBuffer *RenderBuffer, delayPartitions int, hasDelay bool) {
	if !hasDelay {
		r.narrowBandCounters = [FftLengthBy2 - 1]int{}
	} else {
		var channelCounters [FftLengthBy2 - 1]int
		X2 := renderBuffer.Spectrum(delayPartitions)
		for ch := range X2 {
			x := &X2[ch]
			for k := 1; k < FftLengthBy2; k++ {
				if x[k] > 3*stdMax(x[k-1], x[k+1]) {
					channelCounters[k-1]++
				}
			}
		}
		for k := 1; k < FftLengthBy2; k++ {
			if channelCounters[k-1] > 0 {
				r.narrowBandCounters[k-1]++
			} else {
				r.narrowBandCounters[k-1] = 0
			}
		}
	}

	if r.hasNarrowPeakBand {
		r.narrowPeakCounter++
		if r.narrowPeakCounter > r.strongPeakFreezeDuration {
			r.hasNarrowPeakBand = false
			r.narrowPeakBand = 0
		}
	}
	xLatest := renderBuffer.GetBlock(0)
	var maxPeakLevel float32
	for channel := 0; channel < xLatest.NumChannels(); channel++ {
		X2Latest := &renderBuffer.Spectrum(0)[channel]
		peakBin := 0
		for k := 1; k < FftLengthBy2Plus1; k++ {
			if X2Latest[k] > X2Latest[peakBin] {
				peakBin = k
			}
		}
		var nonPeakPower float32
		for k := max(0, peakBin-14); k < peakBin-4; k++ {
			nonPeakPower = stdMax(X2Latest[k], nonPeakPower)
		}
		for k := peakBin + 5; k < min(peakBin+15, FftLengthBy2Plus1); k++ {
			nonPeakPower = stdMax(X2Latest[k], nonPeakPower)
		}
		maxAbs := afirMaxAbsMinMax(xLatest.View(0, channel))
		if xLatest.NumBands() > 1 {
			maxAbs = stdMax(maxAbs, afirMaxAbsMinMax(xLatest.View(1, channel)))
		}
		peakLevel := X2Latest[peakBin]
		if peakBin > 0 && maxAbs > 100 && peakLevel > 100*nonPeakPower {
			if peakLevel > maxPeakLevel {
				maxPeakLevel = peakLevel
				r.narrowPeakBand = peakBin
				r.hasNarrowPeakBand = true
				r.narrowPeakCounter = 0
			}
		}
	}
}

// rsaTestSpectrum draws a power spectrum with narrow peaks, plateaus (ties),
// signed zeros, infinities and NaNs.
func rsaTestSpectrum(rng *rand.Rand, x *[FftLengthBy2Plus1]float32) {
	level := float32(math.Pow(10, 6*rng.Float64()))
	for k := range x {
		x[k] = level * float32(rng.ExpFloat64())
	}
	for range rng.Intn(4) {
		x[rng.Intn(FftLengthBy2Plus1)] *= float32(50 + 1000*rng.Float64())
	}
	if rng.Intn(4) == 0 {
		start := rng.Intn(FftLengthBy2Plus1)
		v := x[start]
		for k := start; k < min(start+1+rng.Intn(20), FftLengthBy2Plus1); k++ {
			x[k] = v
		}
	}
	for k := range x {
		switch rng.Intn(64) {
		case 0:
			x[k] = 0
		case 1:
			x[k] = float32(math.Copysign(0, -1))
		case 2:
			x[k] = float32(math.Inf(1))
		case 3:
			x[k] = float32(math.NaN())
		case 4:
			x[k] = -x[k]
		}
	}
}

func TestRenderSignalAnalyzerMatchesBranchingFormulation(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for trial := 0; trial < 40; trial++ {
		size := 20
		numBands := 1 + rng.Intn(3)
		numChannels := 1 + rng.Intn(3)
		bb := NewBlockBuffer(size, numBands, numChannels)
		sb := NewSpectrumBuffer(size, numChannels)
		rb := NewRenderBuffer(bb, sb, NewFftBuffer(size, numChannels))
		got := &RenderSignalAnalyzer{strongPeakFreezeDuration: 1 + rng.Intn(20)}
		want := &refRenderSignalAnalyzer{strongPeakFreezeDuration: got.strongPeakFreezeDuration}
		for step := 0; step < 400; step++ {
			p := rng.Intn(size)
			sb.Read, bb.Read = p, p
			for i := range sb.Buffer {
				for ch := range sb.Buffer[i] {
					rsaTestSpectrum(rng, &sb.Buffer[i][ch])
				}
			}
			b := &bb.Buffer[p]
			scale := []float32{10, 1000, 30000}[rng.Intn(3)]
			for band := 0; band < numBands; band++ {
				for ch := 0; ch < numChannels; ch++ {
					v := b.View(band, ch)
					for i := range v {
						v[i] = scale * float32(rng.NormFloat64())
					}
				}
			}
			delay, hasDelay := rng.Intn(size), rng.Intn(6) != 0
			got.Update(rb, delay, hasDelay)
			want.Update(rb, delay, hasDelay)
			if got.narrowBandCounters != want.narrowBandCounters || got.narrowPeakBand != want.narrowPeakBand ||
				got.hasNarrowPeakBand != want.hasNarrowPeakBand || got.narrowPeakCounter != want.narrowPeakCounter {
				t.Fatalf("trial %d, step %d: state %+v, want %+v", trial, step, *got, *want)
			}
		}
	}
}

func TestRsaMaxIsStdMax(t *testing.T) {
	vals := []float32{0, float32(math.Copysign(0, -1)), 1, -1, 3, float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), math.Float32frombits(0x7fc00001), math.Float32frombits(1), math.MaxFloat32}
	for _, a := range vals {
		for _, b := range vals {
			if got, want := rsaMax(a, b), stdMax(a, b); math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("rsaMax(%v, %v) = %#x, want %#x", a, b, math.Float32bits(got), math.Float32bits(want))
			}
		}
	}
}

func BenchmarkRenderSignalAnalyzer(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	const size = 1024
	bb := NewBlockBuffer(size, 1, 1)
	sb := NewSpectrumBuffer(size, 1)
	rb := NewRenderBuffer(bb, sb, NewFftBuffer(size, 1))
	for i := range sb.Buffer {
		x := &sb.Buffer[i][0]
		level := float32(1000)
		for k := range x {
			x[k] = level * float32(rng.ExpFloat64())
		}
		x[rng.Intn(FftLengthBy2Plus1)] *= 300
		v := bb.Buffer[i].View(0, 0)
		for k := range v {
			v[k] = 3000 * float32(rng.NormFloat64())
		}
	}
	b.Run("branching", func(b *testing.B) {
		r := &refRenderSignalAnalyzer{strongPeakFreezeDuration: 13}
		for i := 0; i < b.N; i++ {
			sb.Read, bb.Read = i%size, i%size
			r.Update(rb, 3, true)
		}
	})
	b.Run("select", func(b *testing.B) {
		r := &RenderSignalAnalyzer{strongPeakFreezeDuration: 13}
		for i := 0; i < b.N; i++ {
			sb.Read, bb.Read = i%size, i%size
			r.Update(rb, 3, true)
		}
	})
}

func BenchmarkRsaCountNarrowBands(b *testing.B) {
	rng := rand.New(rand.NewSource(3))
	xs := make([][FftLengthBy2Plus1]float32, 1024)
	for i := range xs {
		for k := range xs[i] {
			xs[i][k] = 1000 * float32(rng.ExpFloat64())
		}
		xs[i][rng.Intn(FftLengthBy2Plus1)] *= 300
	}
	var sink int
	b.Run("branching", func(b *testing.B) {
		var c [FftLengthBy2 - 1]int
		for i := 0; i < b.N; i++ {
			x := &xs[i&1023]
			for k := 1; k < FftLengthBy2; k++ {
				if x[k] > 3*stdMax(x[k-1], x[k+1]) {
					c[k-1]++
				}
			}
		}
		sink += c[3]
	})
	b.Run("select", func(b *testing.B) {
		var c [FftLengthBy2 - 1]int
		for i := 0; i < b.N; i++ {
			rsaCountNarrowBands(&xs[i&1023], &c)
		}
		sink += c[3]
	})
	if sink == 1 {
		b.Log(sink)
	}
}
