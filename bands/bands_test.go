package bands

import "testing"

func TestBandsBandSplitterAllocs(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		s := NewBandSplitter(rate, 2)
		n := rate / 100
		data := [][]float32{make([]float32, n), make([]float32, n)}
		bnd := make([][][]float32, s.NumBands())
		for b := range bnd {
			bnd[b] = [][]float32{make([]float32, 160), make([]float32, 160)}
		}
		if a := testing.AllocsPerRun(100, func() {
			s.SplitIntoFrequencyBands(data, bnd)
			s.MergeFrequencyBands(bnd, data)
		}); a != 0 {
			t.Fatalf("rate %d: %v allocs per frame", rate, a)
		}
	}
}
