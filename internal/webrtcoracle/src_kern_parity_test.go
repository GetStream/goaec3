//go:build webrtc_src_oracle && !webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// srcKernLanes returns the lanes whose C++ kernels this machine can run:
// scalar, SSE2 and AVX2 (FMA reported, or GOAEC3_FORCE_FMA=1) on amd64,
// NEON on arm64 (not under purego).
func srcKernLanes(t *testing.T) []aec3.Aec3Optimization {
	var lanes []aec3.Aec3Optimization
	for _, l := range []aec3.Aec3Optimization{aec3.OptNone, aec3.OptSse2, aec3.OptAvx2, aec3.OptNeon} {
		if !SrcKernHasLane(l) {
			continue
		}
		if l == aec3.OptAvx2 && !SrcCPUHasFMA() && os.Getenv("GOAEC3_FORCE_FMA") != "1" {
			t.Logf("AVX2 kernels not run: the CPU does not report FMA (GOAEC3_FORCE_FMA=1 forces them)")
			if os.Getenv("GOAEC3_EXPECT_LANE") == "avx2" {
				t.Fatal("GOAEC3_EXPECT_LANE=avx2 but the AVX2 kernels cannot run")
			}
			continue
		}
		lanes = append(lanes, l)
	}
	return lanes
}

// srcGoMatchedFilterCore is the Go kernel of each lane.
func srcGoMatchedFilterCore(lane aec3.Aec3Optimization, xs int, thr, sm float32, x, y, h []float32, upd *bool,
	es *float32, accum bool, acc, scr []float32) {
	switch lane {
	case aec3.OptSse2:
		aec3.MatchedFilterCoreSse2(xs, thr, sm, x, y, h, upd, es, accum, acc, scr)
	case aec3.OptAvx2:
		aec3.MatchedFilterCoreAvx2(xs, thr, sm, x, y, h, upd, es, accum, acc, scr)
	case aec3.OptNeon:
		aec3.MatchedFilterCoreNeon(xs, thr, sm, x, y, h, upd, es, accum, acc, scr)
	default:
		aec3.MatchedFilterCore(xs, thr, sm, x, y, h, upd, es, accum, acc)
	}
}

func srcSpeechLike(rng *rand.Rand, n int, amp float64) []float32 {
	out := make([]float32, n)
	f := 0.02 + 0.2*rng.Float64()
	for i := range out {
		out[i] = float32(amp*(0.6*math.Sin(f*float64(i))+0.4*rng.NormFloat64()) + 0.25)
	}
	return out
}

func srcEqualBits(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", what, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: [%d] %v want %v", what, i, got[i], want[i])
		}
	}
}

// TestSrcMatchedFilterCoreParity compares MatchedFilterCore of every lane
// (MatchedFilterCore_SSE2 / _AVX2 / _NEON and the scalar core) with the Go
// kernel of the same lane, with and without the accumulated error, over
// filter sizes that exercise the vector tails and start indices near the
// circular-buffer wrap.
func TestSrcMatchedFilterCoreParity(t *testing.T) {
	for _, lane := range srcKernLanes(t) {
		t.Run(srcLaneNames[lane], func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(20 + lane)))
			for _, accum := range []bool{false, true} {
				// The vector cores require h sizes in multiples of 4 (SSE2,
				// NEON) or 8 (AVX2; 16 with the accumulated error); WebRTC
				// uses multiples of 32.
				hSizes := []int{512, 256, 48, 16}
				switch {
				case accum:
					hSizes = []int{512, 256, 64, 32}
				case lane == aec3.OptNone:
					hSizes = []int{512, 256, 37, 4}
				}
				for _, hSize := range hSizes {
					for _, xSize := range []int{hSize + 3, 2*hSize + 1, 1012, 2000} {
						x := srcSpeechLike(rng, xSize, 5000)
						hGo, hC := make([]float32, hSize), make([]float32, hSize)
						accGo, accC := make([]float32, hSize/4), make([]float32, hSize/4)
						scrGo, scrC := make([]float32, hSize), make([]float32, hSize)
						thr := float32(hSize) * 150 * 150
						for call := 0; call < 60; call++ {
							y := srcSpeechLike(rng, 16, 5000)
							if call%17 == 3 {
								y[5] = 32000
							}
							xs := rng.Intn(xSize)
							if call%3 == 0 {
								xs = xSize - 1 - rng.Intn(min(hSize, 20))
							}
							sm := float32(0.7)
							var updGo, updC bool
							var esGo, esC float32
							srcGoMatchedFilterCore(lane, xs, thr, sm, x, y, hGo, &updGo, &esGo, accum, accGo, scrGo)
							SrcMatchedFilterCore(lane, xs, thr, sm, x, y, hC, &updC, &esC, accum, accC, scrC)
							what := fmt.Sprintf("accum=%v h=%d x=%d call %d", accum, hSize, xSize, call)
							srcEqualBits(t, what+" h", hGo, hC)
							if math.Float32bits(esGo) != math.Float32bits(esC) || updGo != updC {
								t.Fatalf("%s: errorSum/updated %v/%v want %v/%v", what, esGo, updGo, esC, updC)
							}
							if accum {
								srcEqualBits(t, what+" accumulated error", accGo, accC)
							}
						}
					}
				}
			}
		})
	}
}

type srcAfirKernels struct {
	apply func(*aec3.RenderBuffer, int, [][]aec3.FftData, *aec3.FftData)
	adapt func(*aec3.RenderBuffer, *aec3.FftData, int, [][]aec3.FftData)
	fr    func(int, [][]aec3.FftData, [][aec3.FftLengthBy2Plus1]float32)
	erl   func([][aec3.FftLengthBy2Plus1]float32, []float32)
}

func srcGoAfirKernels(lane aec3.Aec3Optimization) srcAfirKernels {
	switch lane {
	case aec3.OptSse2:
		return srcAfirKernels{aec3.ApplyFilterSse2, aec3.AdaptPartitionsSse2, aec3.ComputeFrequencyResponseSse2,
			aec3.ErlComputerSSE2}
	case aec3.OptAvx2:
		return srcAfirKernels{aec3.ApplyFilterAvx2, aec3.AdaptPartitionsAvx2, aec3.ComputeFrequencyResponseAvx2,
			aec3.ErlComputerAVX2}
	case aec3.OptNeon:
		return srcAfirKernels{aec3.ApplyFilterNeon, aec3.AdaptPartitionsNeon, aec3.ComputeFrequencyResponseNeon,
			aec3.ErlComputerNEON}
	}
	return srcAfirKernels{aec3.ApplyFilter, aec3.AdaptPartitions, aec3.ComputeFrequencyResponse, aec3.ErlComputer}
}

func srcRandFft(rng *rand.Rand, d *aec3.FftData, scale float64) {
	for k := range d.Re {
		d.Re[k] = float32(rng.NormFloat64() * scale)
		d.Im[k] = float32(rng.NormFloat64() * scale)
	}
}

func srcCmpFft(t *testing.T, what string, got, want *aec3.FftData) {
	t.Helper()
	srcEqualBits(t, what+" re", got.Re[:], want.Re[:])
	srcEqualBits(t, what+" im", got.Im[:], want.Im[:])
}

// TestSrcAdaptiveFilterKernelsParity compares the adaptive filter kernels of
// every lane (ApplyFilter, AdaptPartitions, ComputeFrequencyResponse and
// ErlComputer: _Sse2 / _Avx2 / _Neon and scalar) with the Go kernels of the
// same lane, over render buffers with 1-3 channels, all partition counts up
// to the buffer size and random read positions.
func TestSrcAdaptiveFilterKernelsParity(t *testing.T) {
	for _, lane := range srcKernLanes(t) {
		t.Run(srcLaneNames[lane], func(t *testing.T) {
			k := srcGoAfirKernels(lane)
			rng := rand.New(rand.NewSource(int64(30 + lane)))
			for _, numCh := range []int{1, 2, 3} {
				const size = 17
				fb := aec3.NewFftBuffer(size, numCh)
				rb := aec3.NewRenderBuffer(aec3.NewBlockBuffer(size, 1, numCh), aec3.NewSpectrumBuffer(size, numCh), fb)
				c := NewSrcRenderBuffer(size, numCh)
				for iter := 0; iter < 150; iter++ {
					for i := range fb.Size {
						for ch := range fb.At(i) {
							srcRandFft(rng, &fb.At(i)[ch], 3000)
							c.SetFft(i, ch, &fb.At(i)[ch])
						}
					}
					fb.Read = rng.Intn(size)
					c.SetRead(fb.Read)
					P := 1 + rng.Intn(size)
					np := rng.Intn(P + 1)
					H := make([][]aec3.FftData, P)
					for p := range H {
						H[p] = make([]aec3.FftData, numCh)
						for ch := range H[p] {
							srcRandFft(rng, &H[p][ch], 0.05)
						}
					}
					flat := srcFlatH(H)
					var G aec3.FftData
					srcRandFft(rng, &G, 1e-4)
					name := fmt.Sprintf("ch%d iter%d P%d np%d", numCh, iter, P, np)

					var Sg, Sc aec3.FftData
					k.apply(rb, np, H, &Sg)
					SrcApplyFilter(c, lane, np, flat, P, numCh, &Sc)
					srcCmpFft(t, name+" ApplyFilter", &Sg, &Sc)

					fc := append([]aec3.FftData(nil), flat...)
					k.adapt(rb, &G, np, H)
					SrcAdaptPartitions(c, lane, &G, np, fc, P, numCh)
					fg := srcFlatH(H)
					for i := range fg {
						srcCmpFft(t, fmt.Sprintf("%s AdaptPartitions[%d]", name, i), &fg[i], &fc[i])
					}

					H2g := make([][aec3.FftLengthBy2Plus1]float32, P)
					H2c := make([][aec3.FftLengthBy2Plus1]float32, P)
					H2g[0][1], H2c[0][1] = 5, 5
					k.fr(np, H, H2g)
					SrcFreqResp(lane, np, fc, P, numCh, H2c)
					for p := range H2g {
						srcEqualBits(t, fmt.Sprintf("%s FreqResp[%d]", name, p), H2g[p][:], H2c[p][:])
					}

					eg := make([]float32, aec3.FftLengthBy2Plus1)
					ec := make([]float32, aec3.FftLengthBy2Plus1)
					k.erl(H2g[:np], eg)
					SrcErl(lane, H2c[:np], ec)
					srcEqualBits(t, name+" Erl", eg, ec)
				}
				c.Free()
			}
		})
	}
}
