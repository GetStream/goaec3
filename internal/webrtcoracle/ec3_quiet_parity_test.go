//go:build webrtc_oracle

package webrtcoracle

import (
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// ec3QuietSegment is a stretch of far-end signal: a talker scaled by gain,
// or (noise > 0) uniform noise of that peak amplitude in FloatS16 units.
// room selects the echo path (two delays). The noise segments have no
// near-end talker, so that the echo of noise just above the excitation limit
// is enough for the matched filters to follow a change of the echo path.
type ec3QuietSegment struct {
	frames int
	gain   float64
	noise  float64
	room   int
}

// ec3QuietSchedule covers the far-end levels around the matched filter's
// quiet-render skip: loud speech, long digital silence, noise whose
// decimated level lies below, around and above the excitation limit, and
// transitions between quiet and loud that are shorter and longer than the
// low-rate render buffer (0.6 s). The echo path delay changes at the start
// of every noise segment and during silence: a canceller that skipped the
// matched filters on render above the limit would find the new delay later
// than the reference, and one that updated them below the limit earlier.
func ec3QuietSchedule() []ec3QuietSegment {
	s := []ec3QuietSegment{{frames: 250, gain: 1}, {frames: 200, room: 1}, {frames: 150, gain: 1, room: 1}}
	for i, peak := range []float64{40, 120, 200, 260, 300, 340, 380, 430, 500, 700, 1000, 2000} {
		s = append(s, ec3QuietSegment{frames: 130, noise: peak, room: i % 2})
	}
	s = append(s, ec3QuietSegment{frames: 100, gain: 1})
	// Quiet/loud alternation: 0.2 s, 0.7 s and 1.5 s per state.
	for _, frames := range []int{20, 70, 150} {
		for rep := 0; rep < 3; rep++ {
			s = append(s, ec3QuietSegment{frames: frames, room: 1}, ec3QuietSegment{frames: frames, gain: 1, room: 1},
				ec3QuietSegment{frames: frames, noise: 60}, ec3QuietSegment{frames: frames, gain: 0.02})
		}
	}
	// A slow fade through the limit in both directions.
	for i := 0; i <= 40; i++ {
		s = append(s, ec3QuietSegment{frames: 8, noise: 1500 * math.Pow(0.93, float64(i)), room: 1})
	}
	for i := 40; i >= 0; i-- {
		s = append(s, ec3QuietSegment{frames: 8, noise: 1500 * math.Pow(0.93, float64(i))})
	}
	return append(s, ec3QuietSegment{frames: 100, gain: 1, room: 1})
}

// TestEc3QuietRenderParity runs EchoCanceller3 end to end over
// ec3QuietSchedule and requires bit-exact output, linear output and metrics
// on every frame, and that the scenario both takes the matched filter's
// quiet-render skip and runs the filters.
func TestEc3QuietRenderParity(t *testing.T) {
	type tc struct {
		name    string
		rate    int
		nr, nc  int
		mutate  func(cfg *aec3.EchoCanceller3Config)
		minSkip int
	}
	cases := []tc{
		{"16000Hz_r1_c1", 16000, 1, 1, nil, 300},
		{"48000Hz_r2_c2", 48000, 2, 2, nil, 300},
		{"32000Hz_r1_c1_ds8", 32000, 1, 1, func(cfg *aec3.EchoCanceller3Config) {
			cfg.Delay.DownSamplingFactor = 8
			cfg.Delay.NumFilters = 4
		}, 100},
		{"16000Hz_r1_c1_nopreecho", 16000, 1, 1, func(cfg *aec3.EchoCanceller3Config) {
			cfg.Delay.DetectPreEcho = false
		}, 300},
		{"16000Hz_r1_c2_limit40", 16000, 1, 2, func(cfg *aec3.EchoCanceller3Config) {
			cfg.RenderLevels.PoorExcitationRenderLimit = 40
		}, 100},
		{"16000Hz_r1_c1_limit400", 16000, 1, 1, func(cfg *aec3.EchoCanceller3Config) {
			cfg.RenderLevels.PoorExcitationRenderLimit = 400
		}, 300},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := aec3.DefaultEchoCanceller3Config()
			cfg.Filter.ExportLinearAecOutput = true
			if c.mutate != nil {
				c.mutate(&cfg)
			}
			cpp := NewEc3Canceller(&cfg, nil, c.rate, c.nr, c.nc, ec3Scalar)
			defer cpp.Free()
			g := newEc3GoCanceller(&cfg, nil, c.rate, c.nr, c.nc)

			n := c.rate / 100
			rng := rand.New(rand.NewSource(77))
			talker := newEc3Talker(3, c.rate, 120, 1, 1)
			nearEnd := newEc3Talker(53, c.rate, 210, 0.3, 2.3)
			var rooms [2][][]*ec3Room // [echo path][capture][render]
			for p, delayMs := range []int{80, 136} {
				for cc := 0; cc < c.nc; cc++ {
					var rr []*ec3Room
					for r := 0; r < c.nr; r++ {
						rr = append(rr, newEc3Room(int64(10*cc+r), c.rate, delayMs*c.rate/1000+3*cc+r, 0.4/float64(c.nr)))
					}
					rooms[p] = append(rooms[p], rr)
				}
			}
			renderS16 := ec3Frame(c.nr, n)
			captureS16 := ec3Frame(c.nc, n)
			var flat, cppCap []float32
			cppLin := make([]float32, c.nc*160)
			skipsBefore := aec3.MatchedFilterQuietSkipsForTesting()
			frame := 0
			for _, seg := range ec3QuietSchedule() {
				for f := 0; f < seg.frames; f++ {
					for k := 0; k < n; k++ {
						x := seg.gain * 0.5 * talker.next() * 32768
						if seg.noise > 0 {
							x = seg.noise * (2*rng.Float64() - 1)
						}
						near := 0.3*nearEnd.next()*32768 + 60*rng.NormFloat64()
						if seg.noise > 0 {
							near = 2 * rng.NormFloat64()
						}
						for r := 0; r < c.nr; r++ {
							renderS16[r][k] = float32(x)
						}
						for cc := 0; cc < c.nc; cc++ {
							var y float64
							for r := 0; r < c.nr; r++ {
								// Both echo paths see the render, so each has
								// its history when the schedule selects it.
								for p := range rooms {
									if e := rooms[p][cc][r].next(x); p == seg.room {
										y += e
									}
								}
							}
							captureS16[cc][k] = float32(max(-32768, min(32767, y+near)))
						}
					}
					flat = ec3Flat(renderS16, flat)
					cpp.AnalyzeRender(flat)
					g.analyzeRender(renderS16)
					cppCap = ec3Flat(captureS16, cppCap)
					cpp.ProcessCapture(cppCap, cppLin, false)
					g.processCapture(captureS16)
					for ch := range captureS16 {
						for k, v := range captureS16[ch] {
							if w := cppCap[ch*n+k]; math.Float32bits(v) != math.Float32bits(w) {
								t.Fatalf("frame %d ch %d sample %d: %v want %v", frame, ch, k, v, w)
							}
						}
						for k, v := range g.linear[ch] {
							if w := cppLin[ch*160+k]; math.Float32bits(v) != math.Float32bits(w) {
								t.Fatalf("frame %d linear ch %d sample %d: %v want %v", frame, ch, k, v, w)
							}
						}
					}
					gm, cm := g.ec.GetMetrics(), cpp.Metrics()
					if math.Float64bits(gm.EchoReturnLoss) != math.Float64bits(cm.EchoReturnLoss) ||
						math.Float64bits(gm.EchoReturnLossEnhancement) != math.Float64bits(cm.EchoReturnLossEnhancement) ||
						gm.DelayMs != cm.DelayMs {
						t.Fatalf("frame %d: metrics %+v want %+v", frame, gm, cm)
					}
					frame++
				}
			}
			skips := int(aec3.MatchedFilterQuietSkipsForTesting() - skipsBefore)
			blocks := frame * 160 / aec3.BlockSize
			t.Logf("%d of %d blocks took the quiet-render skip", skips, blocks)
			if skips < c.minSkip || blocks-skips < 300 {
				t.Errorf("%d of %d blocks skipped: the scenario must both skip and run the filters", skips, blocks)
			}
		})
	}
}
