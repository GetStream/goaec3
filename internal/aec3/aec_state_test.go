package aec3

import (
	"math"
	"math/rand"
	"testing"
)

type stTestRig struct {
	cfg    EchoCanceller3Config
	blocks *BlockBuffer
	spec   *SpectrumBuffer
	ffts   *FftBuffer
	rb     *RenderBuffer
	aec    *AecState
	res    *ResidualEchoEstimator
	H2     [][][FftLengthBy2Plus1]float32
	h      [][]float32
	E2, Y2 [][FftLengthBy2Plus1]float32
	R2, Ru [][FftLengthBy2Plus1]float32
	so     []SubtractorOutput
	ext    DelayEstimate
	rng    *rand.Rand
	write  int
}

func newStTestRig(cfg EchoCanceller3Config, numRender, numCapture int) *stTestRig {
	const size = 40
	r := &stTestRig{cfg: cfg, rng: rand.New(rand.NewSource(1))}
	r.blocks = NewBlockBuffer(size, 1, numRender)
	r.spec = NewSpectrumBuffer(size, numRender)
	r.ffts = NewFftBuffer(size, numRender)
	r.rb = NewRenderBuffer(r.blocks, r.spec, r.ffts)
	r.aec = NewAecState(&r.cfg, numCapture)
	r.res = NewResidualEchoEstimator(&r.cfg, numRender)
	lb := cfg.Filter.Refined.LengthBlocks
	r.H2 = make([][][FftLengthBy2Plus1]float32, numCapture)
	r.h = make([][]float32, numCapture)
	for ch := range r.H2 {
		r.H2[ch] = make([][FftLengthBy2Plus1]float32, lb)
		r.h[ch] = make([]float32, lb*FftLengthBy2)
		r.h[ch][2*FftLengthBy2+5] = 0.5
		for i := 2*FftLengthBy2 + 6; i < len(r.h[ch]); i++ {
			r.h[ch][i] = float32(0.1 * math.Pow(0.6, float64(i)/FftLengthBy2) * r.rng.NormFloat64())
		}
		for b := range r.H2[ch] {
			for k := range r.H2[ch][b] {
				r.H2[ch][b][k] = float32(0.01 * math.Pow(0.6, float64(b)))
			}
		}
	}
	r.E2 = make([][FftLengthBy2Plus1]float32, numCapture)
	r.Y2 = make([][FftLengthBy2Plus1]float32, numCapture)
	r.R2 = make([][FftLengthBy2Plus1]float32, numCapture)
	r.Ru = make([][FftLengthBy2Plus1]float32, numCapture)
	r.so = make([]SubtractorOutput, numCapture)
	r.ext = DelayEstimate{Quality: DelayQualityRefined, Delay: 5}
	return r
}

func (r *stTestRig) step() {
	size := r.spec.Size
	r.write = (r.write - 1 + size) % size
	for ch := 0; ch < r.blocks.At(0).NumChannels(); ch++ {
		v := r.blocks.At(r.write).View(0, ch)
		for i := range v {
			v[i] = float32(1000 * r.rng.NormFloat64())
		}
		s := &r.spec.At(r.write)[ch]
		for k := range s {
			s[k] = float32(1.28e8 * r.rng.ExpFloat64())
		}
	}
	read := (r.write + 5) % size
	r.blocks.Write, r.spec.Write, r.ffts.Write = r.write, r.write, r.write
	r.blocks.Read, r.spec.Read, r.ffts.Read = read, read, read
	for ch := range r.Y2 {
		for k := range r.Y2[ch] {
			r.Y2[ch][k] = float32(3e7 * r.rng.ExpFloat64())
			r.E2[ch][k] = r.Y2[ch][k] * 0.1
		}
		r.so[ch].Y2 = 1e7
		r.so[ch].E2RefinedSum = 1e6
		r.so[ch].E2CoarseSum = 2e6
	}
	r.aec.Update(&r.ext, r.H2, r.h, r.rb, r.E2, r.Y2, r.so)
	r.res.Estimate(r.aec, r.rb, r.E2, r.Y2, false, r.R2, r.Ru)
}

func TestAecStateUpdateAllocs(t *testing.T) {
	cfgs := []EchoCanceller3Config{DefaultEchoCanceller3Config(), DefaultEchoCanceller3Config()}
	cfgs[1].Erle.NumSections = 4
	cfgs[1].EpStrength.DefaultLen = -0.8
	cfgs[1].EchoAudibility.UseStationarityProperties = true
	for _, cfg := range cfgs {
		for _, ch := range [][2]int{{1, 1}, {2, 2}} {
			r := newStTestRig(cfg, ch[0], ch[1])
			for i := 0; i < 1500; i++ {
				r.step()
			}
			if a := testing.AllocsPerRun(200, r.step); a != 0 {
				t.Fatalf("AecState.Update + ResidualEchoEstimator.Estimate allocate %v per block", a)
			}
			if !r.aec.UsableLinearEstimate() {
				t.Fatalf("expected a usable linear estimate after warm-up")
			}
		}
	}
}

func TestSignalDependentSectionBoundaries(t *testing.T) {
	// The boundaries cover [headroom, numBlocks) in order.
	for numBlocks := 2; numBlocks <= 40; numBlocks++ {
		for sections := 1; sections <= numBlocks/2; sections++ {
			b := stSetSectionsBoundaries(1, numBlocks, sections)
			if b[len(b)-1] != numBlocks {
				t.Fatalf("blocks %d sections %d: %v", numBlocks, sections, b)
			}
			for i := 1; i < len(b); i++ {
				if b[i] < b[i-1] {
					t.Fatalf("blocks %d sections %d: not monotonic %v", numBlocks, sections, b)
				}
			}
		}
	}
}
