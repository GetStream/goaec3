//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

const (
	subtRenderSize = 40
	subtReadLag    = 1
	subtMaxEcho    = 1024
)

// subtScenario describes one Subtractor parity run.
type subtScenario struct {
	name                  string
	cfg                   aec3.EchoCanceller3Config
	numRender, numCapture int
	blocks                int
	followTransition      bool
	exitAt                int // forced ExitInitialState block (0 = none)
	gainChangeAt          int // echo path gain change block (0 = none)
	delayChangeAt         int // echo path delay change block (0 = none)
	saturateFrom          int // first block of a clipped capture segment (0 = none)
	seed                  int64
}

// subtEnv generates render and capture signals and mirrors the render ring
// into a Go RenderBuffer and the C++ simulation.
type subtEnv struct {
	sc       *subtScenario
	rng      *rand.Rand
	bb       *aec3.BlockBuffer
	sb       *aec3.SpectrumBuffer
	fb       *aec3.FftBuffer
	rb       *aec3.RenderBuffer
	fft      aec3.Aec3Fft
	old      [][]float32
	hist     [][]float64 // render history per channel, newest last
	h        [][][]float64
	echoGain float64
	t        int
	capture  aec3.Block
	flat     []float32
	Y2       [][aec3.FftLengthBy2Plus1]float32
}

func newSubtEnv(sc *subtScenario) *subtEnv {
	e := &subtEnv{sc: sc, rng: rand.New(rand.NewSource(sc.seed)),
		bb:       aec3.NewBlockBuffer(subtRenderSize, 1, sc.numRender),
		sb:       aec3.NewSpectrumBuffer(subtRenderSize, sc.numRender),
		fb:       aec3.NewFftBuffer(subtRenderSize, sc.numRender),
		echoGain: 1,
		capture:  aec3.NewBlock(1, sc.numCapture),
		flat:     make([]float32, sc.numCapture*aec3.BlockSize),
		Y2:       make([][aec3.FftLengthBy2Plus1]float32, sc.numCapture),
	}
	e.rb = aec3.NewRenderBuffer(e.bb, e.sb, e.fb)
	e.old = make([][]float32, sc.numRender)
	e.hist = make([][]float64, sc.numRender)
	for ch := range e.old {
		e.old[ch] = make([]float32, aec3.BlockSize)
		e.hist[ch] = make([]float64, subtMaxEcho+aec3.BlockSize)
	}
	e.newEchoPath(subtReadLag*aec3.BlockSize + 40)
	return e
}

// newEchoPath draws decaying random impulse responses [capture][render] with
// the given bulk delay in samples.
func (e *subtEnv) newEchoPath(delay int) {
	e.h = make([][][]float64, e.sc.numCapture)
	for c := range e.h {
		e.h[c] = make([][]float64, e.sc.numRender)
		for r := range e.h[c] {
			h := make([]float64, delay+200)
			for i := delay; i < len(h); i++ {
				h[i] = e.rng.NormFloat64() * 0.5 * math.Exp(-float64(i-delay)/30) / float64(e.sc.numRender)
			}
			h[delay] += 0.6 / float64(e.sc.numRender)
			e.h[c][r] = h
		}
	}
}

// step generates the next block, writes the render ring and the capture.
// It returns whether the capture is saturated.
func (e *subtEnv) step(sim *SubtSim) bool {
	e.t++
	pos := (e.fb.Write - 1 + subtRenderSize) % subtRenderSize
	seg := (e.t / 150) % 5
	amp := []float64{3000, 9000, 100, 6000, 15000}[seg]
	for ch := 0; ch < e.sc.numRender; ch++ {
		hs := e.hist[ch]
		copy(hs, hs[aec3.BlockSize:])
		x := e.bb.At(pos).View(0, ch)
		for i := range x {
			n := e.t*aec3.BlockSize + i
			mod := 0.6 + 0.4*math.Sin(2*math.Pi*float64(n)/(3000+500*float64(ch)))
			v := amp * mod * e.rng.NormFloat64()
			if seg == 3 {
				v += 4000 * math.Sin(2*math.Pi*float64(n)*13/128)
			}
			v = math.Max(-32768, math.Min(32767, v))
			x[i] = float32(v)
			hs[len(hs)-aec3.BlockSize+i] = float64(x[i])
		}
		X := &e.fb.At(pos)[ch]
		e.fft.PaddedFft(x, e.old[ch], aec3.WindowRectangular, X)
		copy(e.old[ch], x)
		X.Spectrum(aec3.OptNone, e.sb.At(pos)[ch][:])
		sim.SetRender(pos, ch, x, X, &e.sb.At(pos)[ch])
	}
	read := (pos + subtReadLag) % subtRenderSize
	e.fb.Write, e.sb.Write, e.bb.Write = pos, pos, pos
	e.fb.Read, e.sb.Read, e.bb.Read = read, read, read
	sim.SetIndices(pos, read)

	saturate := e.sc.saturateFrom > 0 && e.t >= e.sc.saturateFrom && e.t < e.sc.saturateFrom+60
	saturated := false
	for c := 0; c < e.sc.numCapture; c++ {
		y := e.capture.View(0, c)
		for i := range y {
			var v float64
			for r := 0; r < e.sc.numRender; r++ {
				hs := e.hist[r]
				idx := len(hs) - aec3.BlockSize + i
				for j, hj := range e.h[c][r] {
					if hj != 0 {
						v += hj * hs[idx-j]
					}
				}
			}
			v = v*e.echoGain + e.rng.NormFloat64()*20
			if saturate {
				v *= 8
			}
			if v >= 32767 || v <= -32768 {
				saturated = true
			}
			v = math.Max(-32768, math.Min(32767, v))
			y[i] = float32(v)
		}
		copy(e.flat[c*aec3.BlockSize:], y)
		var Y aec3.FftData
		e.fft.ZeroPaddedFft(y, aec3.WindowRectangular, &Y)
		Y.Spectrum(aec3.OptNone, e.Y2[c][:])
	}
	return saturated
}

func subtCmp(t *testing.T, blk int, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("block %d %s: len go %d c++ %d", blk, what, len(got), len(want))
	}
	for k := range got {
		if math.Float32bits(got[k]) != math.Float32bits(want[k]) {
			t.Fatalf("block %d %s[%d]: go %v c++ %v", blk, what, k, got[k], want[k])
		}
	}
}

func subtCmpOutput(t *testing.T, blk, ch int, g, w *aec3.SubtractorOutput) {
	t.Helper()
	p := fmt.Sprintf("ch%d ", ch)
	subtCmp(t, blk, p+"s_refined", g.SRefined[:], w.SRefined[:])
	subtCmp(t, blk, p+"s_coarse", g.SCoarse[:], w.SCoarse[:])
	subtCmp(t, blk, p+"e_refined", g.ERefined[:], w.ERefined[:])
	subtCmp(t, blk, p+"e_coarse", g.ECoarse[:], w.ECoarse[:])
	subtCmp(t, blk, p+"E_refined.re", g.ERefinedFft.Re[:], w.ERefinedFft.Re[:])
	subtCmp(t, blk, p+"E_refined.im", g.ERefinedFft.Im[:], w.ERefinedFft.Im[:])
	subtCmp(t, blk, p+"E2_refined", g.E2Refined[:], w.E2Refined[:])
	subtCmp(t, blk, p+"E2_coarse", g.E2Coarse[:], w.E2Coarse[:])
	subtCmp(t, blk, p+"scalars",
		[]float32{g.S2Refined, g.S2Coarse, g.E2RefinedSum, g.E2CoarseSum, g.Y2, g.SRefinedMaxAbs, g.SCoarseMaxAbs},
		[]float32{w.S2Refined, w.S2Coarse, w.E2RefinedSum, w.E2CoarseSum, w.Y2, w.SRefinedMaxAbs, w.SCoarseMaxAbs})
}

func subtRun(t *testing.T, sc subtScenario, neon bool) {
	opt := aec3.OptNone
	if neon {
		opt = aec3.OptNeon
	}
	cfg := sc.cfg
	sim := NewSubtSim(&cfg, subtRenderSize, sc.numRender, sc.numCapture, neon, purego)
	defer sim.Free()
	env := newSubtEnv(&sc)

	sub := aec3.NewSubtractor(&cfg, sc.numRender, sc.numCapture, opt)
	rsa := aec3.NewRenderSignalAnalyzer(&cfg)
	aec := aec3.NewAecState(&cfg, sc.numCapture)
	outs := make([]aec3.SubtractorOutput, sc.numCapture)
	wants := make([]aec3.SubtractorOutput, sc.numCapture)
	for i := range outs {
		outs[i].Reset()
	}
	E2 := make([][aec3.FftLengthBy2Plus1]float32, sc.numCapture)
	maxBlocks := max(cfg.Filter.Refined.LengthBlocks, cfg.Filter.RefinedInitial.LengthBlocks)
	wantH2 := make([][aec3.FftLengthBy2Plus1]float32, maxBlocks)
	wantH := make([]float32, aec3.GetTimeDomainLength(maxBlocks))

	transitions := 0
	for blk := 1; blk <= sc.blocks; blk++ {
		var v aec3.EchoPathVariability
		if blk == sc.gainChangeAt {
			env.echoGain = 0.05
			v.GainChange = true
		}
		if blk == sc.delayChangeAt {
			env.newEchoPath(subtReadLag*aec3.BlockSize + 150)
			v.DelayChange = aec3.DelayAdjustmentNewDetectedDelay
		}
		saturated := env.step(sim)
		if v.AudioPathChanged() {
			sim.HandleEchoPathChange(v)
			sub.HandleEchoPathChange(v)
			aec.HandleEchoPathChange(v)
		}
		if blk == sc.exitAt {
			sim.ExitInitialState()
			sub.ExitInitialState()
		}

		sim.Process(env.flat, env.Y2, saturated, sc.followTransition, wants)

		aec.UpdateCaptureSaturation(saturated)
		rsa.Update(env.rb, aec.MinDirectPathFilterDelay(), true)
		if sc.followTransition && aec.TransitionTriggered() {
			sub.ExitInitialState()
			transitions++
		}
		sub.Process(env.rb, &env.capture, rsa, aec, outs)
		for ch := range E2 {
			E2[ch] = outs[ch].E2Refined
		}
		aec.Update(nil, sub.FilterFrequencyResponses(), sub.FilterImpulseResponses(), env.rb, E2, env.Y2, outs)

		for ch := 0; ch < sc.numCapture; ch++ {
			subtCmpOutput(t, blk, ch, &outs[ch], &wants[ch])
			n := sim.FreqResp(ch, wantH2)
			got := sub.FilterFrequencyResponses()[ch]
			if len(got) != n {
				t.Fatalf("block %d ch%d: freq resp len go %d c++ %d", blk, ch, len(got), n)
			}
			for p := range got {
				subtCmp(t, blk, fmt.Sprintf("ch%d H2[%d]", ch, p), got[p][:], wantH2[p][:])
			}
			n = sim.ImpulseResp(ch, wantH)
			subtCmp(t, blk, fmt.Sprintf("ch%d h", ch), sub.FilterImpulseResponses()[ch], wantH[:n])
		}
		ws := sim.AecState()
		gs := SubtAecState{aec.MinDirectPathFilterDelay(), aec.TransitionTriggered(), aec.SaturatedCapture(),
			aec.UsableLinearEstimate(), aec.FullBandErleLog2()}
		if gs != ws {
			t.Fatalf("block %d: aec state go %+v c++ %+v", blk, gs, ws)
		}
	}
	if sc.followTransition && transitions == 0 {
		t.Errorf("no initial-state transition was triggered")
	}
}

func subtScenarios() []subtScenario {
	def := aec3.DefaultEchoCanceller3Config()

	alt := aec3.DefaultEchoCanceller3Config()
	alt.Filter.Refined.LengthBlocks = 20
	alt.Filter.RefinedInitial.LengthBlocks = 8
	alt.Filter.Coarse.LengthBlocks = 15
	alt.Filter.CoarseInitial.LengthBlocks = 10
	alt.Filter.ConfigChangeDurationBlocks = 100
	alt.Filter.CoarseResetHangoverBlocks = 5
	alt.Filter.ConservativeInitialPhase = true
	alt.Filter.EnableCoarseFilterOutputUsage = false
	alt.Filter.InitialStateSeconds = 1.5

	short := aec3.DefaultEchoCanceller3Config()
	short.Filter.Refined.LengthBlocks = 10 // ReverbDecayEstimator needs at least 9
	short.Filter.RefinedInitial.LengthBlocks = 6
	short.Filter.Coarse.LengthBlocks = 14
	short.Filter.CoarseInitial.LengthBlocks = 12
	short.Filter.Refined.LeakageDiverged = 0.3
	short.Filter.Coarse.Rate = 0.5
	short.Filter.CoarseResetHangoverBlocks = 0

	return []subtScenario{
		{name: "default_1x1", cfg: def, numRender: 1, numCapture: 1, blocks: 4000, followTransition: true,
			gainChangeAt: 1600, delayChangeAt: 2600, saturateFrom: 3300, seed: 1},
		{name: "default_1x2", cfg: def, numRender: 1, numCapture: 2, blocks: 3000, followTransition: true,
			gainChangeAt: 1200, delayChangeAt: 2000, saturateFrom: 2500, seed: 2},
		{name: "default_2x2", cfg: def, numRender: 2, numCapture: 2, blocks: 2500, followTransition: true,
			gainChangeAt: 1000, delayChangeAt: 1800, seed: 3},
		{name: "alt_1x1", cfg: alt, numRender: 1, numCapture: 1, blocks: 3500, exitAt: 700,
			gainChangeAt: 1500, delayChangeAt: 2300, saturateFrom: 3000, seed: 4},
		{name: "alt_1x2_follow", cfg: alt, numRender: 1, numCapture: 2, blocks: 4000, followTransition: true,
			gainChangeAt: 2400, delayChangeAt: 3000, seed: 5},
		{name: "short_refined_1x1", cfg: short, numRender: 1, numCapture: 1, blocks: 3000, exitAt: 300,
			gainChangeAt: 1000, delayChangeAt: 1900, saturateFrom: 2400, seed: 6},
	}
}

func TestSubtSubtractorParity(t *testing.T) {
	for _, sc := range subtScenarios() {
		for _, neon := range []bool{false, true} {
			lane := "none"
			if neon {
				lane = "neon"
			}
			t.Run(sc.name+"_"+lane, func(t *testing.T) { subtRun(t, sc, neon) })
		}
	}
}
