//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/bands"
)

func bandsCheck(t *testing.T, what string, got, want []float32) {
	t.Helper()
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: got %v (%#x) want %v (%#x)", what, i, got[i], math.Float32bits(got[i]),
				want[i], math.Float32bits(want[i]))
		}
	}
}

func bandsCheck16(t *testing.T, what string, got, want []int16) {
	t.Helper()
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]: got %d want %d", what, i, got[i], want[i])
		}
	}
}

// bandsSpeech generates a speech-like signal in [-1, 1] units: a voiced
// harmonic source with a wandering pitch and syllabic envelope, unvoiced noise
// bursts, silences, and occasional overdriven (clipping) segments.
type bandsSpeech struct {
	rng       *rand.Rand
	rate      float64
	phase     float64
	f0        float64
	env       float64
	target    float64
	voiced    bool
	countdown int
	gain      float64
}

func newBandsSpeech(seed int64, rate int) *bandsSpeech {
	return &bandsSpeech{rng: rand.New(rand.NewSource(seed)), rate: float64(rate), f0: 140, gain: 0.3}
}

func (s *bandsSpeech) next() float32 {
	if s.countdown <= 0 {
		s.countdown = int(s.rate * (0.03 + 0.2*s.rng.Float64()))
		s.voiced = s.rng.Intn(3) != 0
		switch r := s.rng.Intn(10); {
		case r == 0:
			s.target = 0
		case r < 3:
			s.target = 1.5 + s.rng.Float64() // overdriven, clips at +-1
		default:
			s.target = 0.05 + 0.6*s.rng.Float64()
		}
		s.f0 = 80 + 250*s.rng.Float64()
	}
	s.countdown--
	s.env += (s.target - s.env) * 0.002
	s.f0 *= 1 + 0.0005*s.rng.NormFloat64()
	s.phase += 2 * math.Pi * s.f0 / s.rate
	var v float64
	if s.voiced {
		for h := 1; h <= 30; h++ {
			if float64(h)*s.f0 > s.rate/2 {
				break
			}
			v += math.Sin(float64(h)*s.phase) / float64(h)
		}
		v *= 0.6
	} else {
		v = s.rng.NormFloat64() * 0.4
	}
	return float32(s.env * v)
}

func TestBandsAudioUtilParity(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const n = 20000
	src := make([]float32, n)
	for i := range src {
		switch i % 5 {
		case 0:
			src[i] = float32(rng.NormFloat64() * 20000)
		case 1:
			src[i] = float32(math.Floor(rng.Float64()*70000-35000)) + 0.5
		case 2:
			src[i] = math.Nextafter32(float32(math.Floor(rng.Float64()*70000-35000))+0.5, float32(math.Inf(int(rng.Intn(2))*2-1)))
		case 3:
			src[i] = float32(rng.NormFloat64() * 1.5)
		default:
			src[i] = float32(rng.Float64()-0.5) * 1e-3
		}
	}
	copy(src, []float32{0, float32(math.Copysign(0, -1)), 0.5, -0.5, 0.49999997, -0.49999997, 32767, 32767.5,
		32768, -32768, -32768.5, -40000, 40000, 1, -1, 1.0000001, -1.0000001})
	g16, c16 := make([]int16, n), make([]int16, n)
	bands.FloatS16ToS16Slice(src, g16)
	BandsFloatS16ToS16(src, c16)
	bandsCheck16(t, "FloatS16ToS16", g16, c16)

	gf, cf := make([]float32, n), make([]float32, n)
	bands.S16ToFloatS16Slice(c16, gf)
	BandsS16ToFloatS16(c16, cf)
	bandsCheck(t, "S16ToFloatS16", gf, cf)

	bands.FloatToFloatS16Slice(src, gf)
	BandsFloatToFloatS16(src, cf)
	bandsCheck(t, "FloatToFloatS16", gf, cf)

	bands.FloatS16ToFloatSlice(src, gf)
	BandsFloatS16ToFloat(src, cf)
	bandsCheck(t, "FloatS16ToFloat", gf, cf)
}

func TestBandsThreeBandFilterBankParity(t *testing.T) {
	g := bands.NewThreeBandFilterBank()
	o := NewBandsThreeBandFilterBank()
	defer o.Free()
	sp := newBandsSpeech(1, 48000)
	rng := rand.New(rand.NewSource(4))
	var in, outG [bands.FullBandSize]float32
	split := make([]float32, 3*160)
	splitC := make([]float32, 3*160)
	outC := make([]float32, bands.FullBandSize)
	for frame := 0; frame < 3000; frame++ {
		for i := range in {
			in[i] = bands.FloatToFloatS16(sp.next())
			if frame%97 == 0 {
				in[i] = float32(rng.NormFloat64() * 30000)
			}
		}
		gb := [bands.NumBands][]float32{split[0:160], split[160:320], split[320:480]}
		g.Analysis(&in, &gb)
		o.Analysis(in[:], splitC)
		bandsCheck(t, fmt.Sprintf("frame %d analysis", frame), split, splitC)

		// Perturb the bands identically before synthesis.
		for i := range split {
			split[i] *= float32(1 + 0.5*rng.Float64())
			splitC[i] = split[i]
		}
		g.Synthesis(&gb, &outG)
		o.Synthesis(splitC, outC)
		bandsCheck(t, fmt.Sprintf("frame %d synthesis", frame), outG[:], outC)
	}
}

func TestBandsQMFParity(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	sp := newBandsSpeech(2, 32000)
	var gs1, gs2, cs1, cs2, gt1, gt2, ct1, ct2 [6]int32
	in := make([]int16, 320)
	gl, gh, cl, ch := make([]int16, 160), make([]int16, 160), make([]int16, 160), make([]int16, 160)
	gout, cout := make([]int16, 320), make([]int16, 320)
	for frame := 0; frame < 5000; frame++ {
		for i := range in {
			in[i] = bands.FloatS16ToS16(bands.FloatToFloatS16(sp.next()))
			if frame%53 == 0 {
				// Full-scale alternating and random extremes.
				if i%2 == 0 {
					in[i] = 32767
				} else {
					in[i] = -32768
				}
				if frame%106 == 0 {
					in[i] = int16(rng.Intn(65536) - 32768)
				}
			}
		}
		bands.AnalysisQMF(in, gl, gh, &gs1, &gs2)
		BandsAnalysisQMF(in, cl, ch, &cs1, &cs2)
		bandsCheck16(t, fmt.Sprintf("frame %d low", frame), gl, cl)
		bandsCheck16(t, fmt.Sprintf("frame %d high", frame), gh, ch)
		if gs1 != cs1 || gs2 != cs2 {
			t.Fatalf("frame %d analysis state mismatch", frame)
		}
		if frame%29 == 0 {
			for i := range gl {
				gl[i] = int16(rng.Intn(65536) - 32768)
				gh[i] = int16(rng.Intn(65536) - 32768)
			}
			copy(cl, gl)
			copy(ch, gh)
		}
		bands.SynthesisQMF(gl, gh, gout, &gt1, &gt2)
		BandsSynthesisQMF(cl, ch, cout, &ct1, &ct2)
		bandsCheck16(t, fmt.Sprintf("frame %d synthesis", frame), gout, cout)
		if gt1 != ct1 || gt2 != ct2 {
			t.Fatalf("frame %d synthesis state mismatch", frame)
		}
	}
}

func TestBandsAudioBufferSplitMergeParity(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		for _, numCh := range []int{1, 2} {
			t.Run(fmt.Sprintf("%d_ch%d", rate, numCh), func(t *testing.T) {
				n := rate / 100
				s := bands.NewBandSplitter(rate, numCh)
				o := NewBandsAudioBuffer(rate, numCh)
				defer o.Free()
				nb := s.NumBands()
				if nb*s.NumFramesPerBand() != n || s.NumFramesPerBand() != 160 {
					t.Fatalf("bad layout %d x %d", nb, s.NumFramesPerBand())
				}
				gens := make([]*bandsSpeech, numCh)
				for ch := range gens {
					gens[ch] = newBandsSpeech(int64(rate+ch), rate)
				}
				rng := rand.New(rand.NewSource(int64(rate * numCh)))

				in := make([]float32, numCh*n)
				dataFlat := make([]float32, numCh*n)
				data := make([][]float32, numCh)
				for ch := range data {
					data[ch] = dataFlat[ch*n : (ch+1)*n]
				}
				bandsFlat := make([]float32, nb*numCh*160)
				bnd := make([][][]float32, nb)
				for b := range bnd {
					bnd[b] = make([][]float32, numCh)
					for ch := range bnd[b] {
						i := (b*numCh + ch) * 160
						bnd[b][ch] = bandsFlat[i : i+160]
					}
				}
				cBands := make([]float32, len(bandsFlat))
				cData := make([]float32, len(dataFlat))

				saturated := 0
				for frame := 0; frame < 2000; frame++ {
					for ch := 0; ch < numCh; ch++ {
						for i := 0; i < n; i++ {
							in[ch*n+i] = gens[ch].next()
						}
					}
					// AudioBuffer::CopyFrom on float data.
					bands.FloatToFloatS16Slice(in, dataFlat)
					o.CopyFrom(in)
					o.ReadData(cData)
					bandsCheck(t, fmt.Sprintf("frame %d CopyFrom", frame), dataFlat, cData)

					if nb > 1 {
						o.Split()
					}
					s.SplitIntoFrequencyBands(data, bnd)
					o.ReadBands(cBands)
					bandsCheck(t, fmt.Sprintf("frame %d split", frame), bandsFlat, cBands)

					// Process the bands identically (gain up to 2x pushes the
					// 32 kHz synthesis into int16 saturation).
					g := float32(0.5 + 1.5*rng.Float64())
					for i := range bandsFlat {
						bandsFlat[i] *= g
					}
					o.WriteBands(bandsFlat)

					if nb > 1 {
						o.Merge()
					}
					s.MergeFrequencyBands(bnd, data)
					o.ReadData(cData)
					bandsCheck(t, fmt.Sprintf("frame %d merge", frame), dataFlat, cData)
					for _, v := range dataFlat {
						if v >= 32767 || v <= -32768 {
							saturated++
						}
					}
				}
				if rate == 32000 && saturated == 0 {
					t.Fatalf("merge never saturated; the int16 clamp is not exercised")
				}
				t.Logf("%d saturated merged samples", saturated)
			})
		}
	}
}
