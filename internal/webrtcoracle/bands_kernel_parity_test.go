//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/bands"
)

// bandsNonFinite are inputs whose NaN payloads and infinities the band
// kernels must propagate exactly like the C++.
var bandsNonFinite = []float32{
	float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
	math.Float32frombits(0x7fa00001), math.Float32frombits(0xffc00123), 0,
	float32(math.Copysign(0, -1)), 1e30, -3e38,
}

// TestBandsThreeBandNonFiniteParity drives the filter bank with frames that
// hold NaNs (with distinct payloads), infinities and overflowing values, and
// checks every output bit against the C++.
func TestBandsThreeBandNonFiniteParity(t *testing.T) {
	for seed := int64(1); seed <= 40; seed++ {
		rng := rand.New(rand.NewSource(seed))
		var g bands.ThreeBandFilterBank
		c := NewBandsThreeBandFilterBank()
		defer c.Free()
		for frame := 0; frame < 50; frame++ {
			// Some frames are clean, so the state carries non-finite values
			// into finite frames and back.
			rate := 30
			if frame%3 == 2 {
				rate = 0
			}
			pick := func() float32 {
				if rate > 0 && rng.Intn(rate) == 0 {
					return bandsNonFinite[rng.Intn(len(bandsNonFinite))]
				}
				return float32(rng.NormFloat64() * 3000)
			}
			in := make([]float32, 480)
			for i := range in {
				in[i] = pick()
			}
			cb := make([]float32, 480)
			c.Analysis(in, cb)
			gb := [3][]float32{make([]float32, 160), make([]float32, 160), make([]float32, 160)}
			g.Analysis((*[480]float32)(in), &gb)
			for b := range 3 {
				bandsCheck(t, fmt.Sprintf("seed %d frame %d analysis band %d", seed, frame, b),
					gb[b], cb[b*160:(b+1)*160])
			}

			for i := range cb {
				cb[i] = pick()
			}
			for b := range 3 {
				copy(gb[b], cb[b*160:])
			}
			cout := make([]float32, 480)
			var gout [480]float32
			c.Synthesis(cb, cout)
			g.Synthesis(&gb, &gout)
			bandsCheck(t, fmt.Sprintf("seed %d frame %d synthesis", seed, frame), gout[:], cout)
		}
	}
}

// TestBandsConversionNonFiniteParity checks the slice conversions on special
// values in every lane position against the C++.
func TestBandsConversionNonFiniteParity(t *testing.T) {
	specials := append([]float32{1, -1, 1.5, -1.5, 32767.5, -32768.5, 32766.5, 40000, -40000,
		0.99999994, 1.0000001, 32768, -32768, 1e-40}, bandsNonFinite...)
	for rot := range specials {
		for _, n := range []int{1, 7, 8, 9, 31, 160, 480} {
			src := make([]float32, n)
			for i := range src {
				src[i] = specials[(i+rot)%len(specials)]
			}
			got := make([]float32, n)
			want := make([]float32, n)
			bands.FloatToFloatS16Slice(src, got)
			BandsFloatToFloatS16(src, want)
			bandsCheck(t, "FloatToFloatS16", got, want)
			bands.FloatS16ToFloatSlice(src, got)
			BandsFloatS16ToFloat(src, want)
			bandsCheck(t, "FloatS16ToFloat", got, want)
			got16 := make([]int16, n)
			want16 := make([]int16, n)
			bands.FloatS16ToS16Slice(src, got16)
			BandsFloatS16ToS16(src, want16)
			bandsCheck16(t, "FloatS16ToS16", got16, want16)
		}
	}
}

// BenchmarkBandsSplitMerge reports ns per 10 ms frame for the float input
// copy (FloatToFloatS16), one band split and one merge: Go BandSplitter
// against webrtc::AudioBuffer CopyFrom, SplitIntoFrequencyBands and
// MergeFrequencyBands.
func BenchmarkBandsSplitMerge(b *testing.B) {
	for _, tc := range []struct{ rate, ch int }{{32000, 1}, {32000, 2}, {48000, 1}, {48000, 2}} {
		n := tc.rate / 100
		rng := rand.New(rand.NewSource(1))
		data := make([]float32, n*tc.ch)
		for i := range data {
			data[i] = float32(rng.NormFloat64() * 0.2)
		}
		name := fmt.Sprintf("%dkHz_%dch", tc.rate/1000, tc.ch)
		b.Run(name+"/go", func(b *testing.B) {
			s := bands.NewBandSplitter(tc.rate, tc.ch)
			full := make([][]float32, tc.ch)
			for ch := range full {
				full[ch] = make([]float32, n)
			}
			split := make([][][]float32, s.NumBands())
			for band := range split {
				split[band] = make([][]float32, tc.ch)
				for ch := range split[band] {
					split[band][ch] = make([]float32, 160)
				}
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for ch := range full {
					bands.FloatToFloatS16Slice(data[ch*n:(ch+1)*n], full[ch])
				}
				s.SplitIntoFrequencyBands(full, split)
				s.MergeFrequencyBands(split, full)
			}
		})
		b.Run(name+"/libwebrtc", func(b *testing.B) {
			a := NewBandsAudioBuffer(tc.rate, tc.ch)
			defer a.Free()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				a.CopyFrom(data)
				a.Split()
				a.Merge()
			}
		})
	}
}
