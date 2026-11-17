package aec3

import (
	"math/rand"
	"testing"
)

func TestSubtractorProcessZeroAlloc(t *testing.T) {
	for _, opt := range []Aec3Optimization{OptNone, DetectOptimization()} {
		cfg := DefaultEchoCanceller3Config()
		// Coarse longer than refined exercises the SpectralSums path.
		cfg.Filter.Coarse.LengthBlocks = 15
		cfg.Filter.CoarseInitial.LengthBlocks = 15
		const size, numRender, numCapture = 40, 1, 2
		bb := NewBlockBuffer(size, 1, numRender)
		sb := NewSpectrumBuffer(size, numRender)
		fb := NewFftBuffer(size, numRender)
		rb := NewRenderBuffer(bb, sb, fb)
		var fft Aec3Fft
		rng := rand.New(rand.NewSource(1))
		old := make([]float32, BlockSize)
		capture := NewBlock(1, numCapture)

		sub := NewSubtractor(&cfg, numRender, numCapture, opt)
		rsa := NewRenderSignalAnalyzer(&cfg)
		aec := NewAecState(&cfg, numCapture)
		outs := make([]SubtractorOutput, numCapture)

		step := func() {
			pos := (fb.Write - 1 + size) % size
			x := bb.Buffer[pos].View(0, 0)
			for i := range x {
				x[i] = float32(rng.NormFloat64() * 2000)
			}
			fft.PaddedFft(x, old, WindowRectangular, &fb.Buffer[pos][0])
			copy(old, x)
			fb.Buffer[pos][0].Spectrum(opt, sb.Buffer[pos][0][:])
			fb.Write, sb.Write, bb.Write = pos, pos, pos
			fb.Read, sb.Read, bb.Read = pos, pos, pos
			for ch := 0; ch < numCapture; ch++ {
				y := capture.View(0, ch)
				for i := range y {
					y[i] = 0.5*x[i] + float32(rng.NormFloat64()*10)
				}
			}
			rsa.Update(rb, 0, true)
			sub.Process(rb, &capture, rsa, aec, outs)
		}
		for i := 0; i < 300; i++ {
			step()
		}
		sub.ExitInitialState()
		if allocs := testing.AllocsPerRun(200, step); allocs != 0 {
			t.Fatalf("opt %v: Process allocates %v times per call", opt, allocs)
		}
	}
}
