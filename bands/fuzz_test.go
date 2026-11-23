package bands

import (
	"encoding/binary"
	"math"
	"testing"
)

// FuzzBandSplitter splits and merges fuzzed full-band FloatS16 frames at
// every rate and channel count. Nothing may panic, and finite input must
// give finite bands and a finite merged signal; the [-1, 1] <-> FloatS16
// conversions must stay within their clamping ranges (NaN aside).
func FuzzBandSplitter(f *testing.F) {
	f.Add([]byte{0, 0, 1, 2, 3, 4})
	f.Add([]byte{1, 1, 0x7f, 0x7f, 0xff, 0xff, 0, 0, 0x80, 0x7f})
	f.Add([]byte{2, 0, 0xff, 0xff, 0x7f, 0x7f, 1, 0, 0, 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 2 {
			return
		}
		rate := []int{16000, 32000, 48000}[int(data[0])%3]
		numChannels := 1 + int(data[1])%4
		raw := data[2:]
		s := NewBandSplitter(rate, numChannels)
		n := rate / 100
		frame := make([][]float32, numChannels)
		for ch := range frame {
			frame[ch] = make([]float32, n)
		}
		bnd := make([][][]float32, s.NumBands())
		for b := range bnd {
			bnd[b] = make([][]float32, numChannels)
			for ch := range bnd[b] {
				bnd[b][ch] = make([]float32, 160)
			}
		}
		pos := 0
		next := func() float32 {
			if len(raw) < 4 {
				return 0
			}
			v := math.Float32frombits(binary.LittleEndian.Uint32(raw[pos%(len(raw)-3):]))
			pos += 4
			return v
		}
		for range 3 {
			finite := true
			for ch := range frame {
				for k := range frame[ch] {
					x := next()
					if math.IsNaN(float64(x)) {
						x = 0
					}
					// [-1, 1] -> FloatS16 clamps to [-32768, 32768].
					y := FloatToFloatS16(x)
					if y < -32768 || y > 32768 {
						t.Fatalf("FloatToFloatS16(%v) = %v", x, y)
					}
					if z := FloatS16ToFloat(x); z < -1 || z > 1 {
						t.Fatalf("FloatS16ToFloat(%v) = %v", x, z)
					}
					frame[ch][k] = y
					finite = finite && !math.IsInf(float64(y), 0)
				}
			}
			if s.NumBands() == 1 {
				continue
			}
			s.SplitIntoFrequencyBands(frame, bnd)
			for b := range bnd {
				for ch := range bnd[b] {
					for _, v := range bnd[b][ch] {
						if finite && (math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)) {
							t.Fatalf("band %d ch %d: %v", b, ch, v)
						}
					}
				}
			}
			s.MergeFrequencyBands(bnd, frame)
			for ch := range frame {
				for _, v := range frame[ch] {
					if finite && (math.IsNaN(float64(v)) || math.IsInf(float64(v), 0)) {
						t.Fatalf("merged ch %d: %v", ch, v)
					}
				}
			}
		}
	})
}
