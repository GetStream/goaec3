package aec3

import (
	"math"
	"math/rand"
	"testing"
)

// erTestRig drives an EchoRemover with a synthetic render ring and echo.
type erTestRig struct {
	rng     *rand.Rand
	bb      *BlockBuffer
	sb      *SpectrumBuffer
	fb      *FftBuffer
	rb      *RenderBuffer
	fft     Aec3Fft
	old     [][]float32
	capture Block
	linear  Block
	er      *EchoRemover
	t       int
}

const erTestRingSize = 30

func newErTestRig(cfg *EchoCanceller3Config, rate, nr, nc int) *erTestRig {
	nb := NumBandsForRate(rate)
	r := &erTestRig{rng: rand.New(rand.NewSource(1)),
		bb: NewBlockBuffer(erTestRingSize, nb, nr), sb: NewSpectrumBuffer(erTestRingSize, nr),
		fb: NewFftBuffer(erTestRingSize, nr), capture: NewBlock(nb, nc), linear: NewBlock(1, nc),
		er: NewEchoRemover(cfg, rate, nr, nc)}
	r.rb = NewRenderBuffer(r.bb, r.sb, r.fb)
	r.old = make([][]float32, nr)
	for ch := range r.old {
		r.old[ch] = make([]float32, BlockSize)
	}
	return r
}

func (r *erTestRig) step(withLinear bool) {
	r.t++
	pos := (r.fb.Write - 1 + erTestRingSize) % erTestRingSize
	blk := &r.bb.Buffer[pos]
	for b := 0; b < blk.NumBands(); b++ {
		for ch := 0; ch < blk.NumChannels(); ch++ {
			x := blk.View(b, ch)
			for i := range x {
				x[i] = float32(3000 * r.rng.NormFloat64())
			}
		}
	}
	for ch := 0; ch < blk.NumChannels(); ch++ {
		x := blk.View(0, ch)
		r.fft.PaddedFft(x, r.old[ch], WindowRectangular, &r.fb.Buffer[pos][ch])
		copy(r.old[ch], x)
		r.fb.Buffer[pos][ch].Spectrum(OptNone, r.sb.Buffer[pos][ch][:])
	}
	read := (pos + 1) % erTestRingSize
	r.fb.Write, r.sb.Write, r.bb.Write = pos, pos, pos
	r.fb.Read, r.sb.Read, r.bb.Read = read, read, read
	prev := &r.bb.Buffer[read]
	for b := 0; b < r.capture.NumBands(); b++ {
		for ch := 0; ch < r.capture.NumChannels(); ch++ {
			y := r.capture.View(b, ch)
			x := prev.View(b, 0)
			for i := range y {
				y[i] = 0.5*x[i] + float32(30*r.rng.NormFloat64())
			}
		}
	}
	var lin *Block
	if withLinear {
		lin = &r.linear
	}
	r.er.ProcessCapture(EchoPathVariability{}, false, nil, r.rb, lin, &r.capture)
}

func TestEchoRemoverProcessCaptureNoAllocs(t *testing.T) {
	for _, tc := range []struct{ rate, nr, nc int }{{16000, 1, 1}, {48000, 2, 2}, {32000, 1, 3}} {
		cfg := DefaultEchoCanceller3Config()
		r := newErTestRig(&cfg, tc.rate, tc.nr, tc.nc)
		for i := 0; i < 50; i++ {
			r.step(true)
		}
		if a := testing.AllocsPerRun(100, func() { r.step(true) }); a != 0 {
			t.Errorf("rate %d %dx%d: %v allocs per ProcessCapture", tc.rate, tc.nr, tc.nc, a)
		}
		r.er.SetCaptureOutputUsage(false)
		if a := testing.AllocsPerRun(100, func() { r.step(false) }); a != 0 {
			t.Errorf("rate %d %dx%d unused output: %v allocs per ProcessCapture", tc.rate, tc.nr, tc.nc, a)
		}
		m := r.er.GetMetrics()
		if math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement) {
			t.Errorf("rate %d: NaN metrics %+v", tc.rate, m)
		}
	}
}

func TestErSignalTransition(t *testing.T) {
	var a, b [BlockSize]float32
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(1000 - i)
	}
	var out [FftLengthBy2]float32
	erSignalTransition(&a, &a, &out)
	if out != a {
		t.Fatalf("identical views must copy")
	}
	erSignalTransition(&a, &b, &out)
	for k := 30; k < BlockSize; k++ {
		if out[k] != b[k] {
			t.Fatalf("out[%d] = %v, want %v", k, out[k], b[k])
		}
	}
	for k := 0; k < 30; k++ {
		w := float32(float32(k+1) * (float32(1) / 31))
		if want := float32(w*b[k]) + float32((1-w)*a[k]); out[k] != want {
			t.Fatalf("out[%d] = %v, want %v", k, out[k], want)
		}
	}
}

func TestEchoRemoverMetricsSchedule(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	a := NewAecState(&cfg, 1)
	m := NewEchoRemoverMetrics()
	var noise, gain [FftLengthBy2Plus1]float32
	reported := 0
	for i := 1; i <= 3*MetricsReportingIntervalBlocks; i++ {
		m.Update(a, &noise, &gain)
		if m.MetricsReported() {
			reported++
			if i%MetricsReportingIntervalBlocks != 0 {
				t.Fatalf("reported at block %d", i)
			}
		}
	}
	if reported != 3 {
		t.Fatalf("reported %d times, want 3", reported)
	}
}

func TestUpdateDbMetric(t *testing.T) {
	var v [FftLengthBy2Plus1]float32
	for i := range v {
		v[i] = float32(i)
	}
	s := [2]DbMetric{{0, 1000, 0}, {0, 1000, 0}}
	UpdateDbMetric(&v, &s)
	// Bands are [0, 32) and [32, 64); averages 15.5 and 47.5.
	if s[0].SumValue != 15.5 || s[1].SumValue != 47.5 || s[0].FloorValue != 15.5 || s[1].CeilValue != 47.5 {
		t.Fatalf("got %+v", s)
	}
}
