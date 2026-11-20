package aec3

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestEc3ViewPerCall runs two cancellers on the same signal: one through a
// view that is bound once (NewEchoCanceller3) and one through a view that is
// bound anew for every call, as the public package does. Their outputs and
// metrics must agree bit for bit: whatever a component keeps between calls
// is in its state, not in its view (a slice of a view that changes its
// length, for instance).
func TestEc3ViewPerCall(t *testing.T) {
	frames := 600
	if testing.Short() {
		frames = 250
	}
	type variant struct {
		name                string
		rate                int
		numRender           int
		numCapture          int
		multichannel        bool
		differentChannels   bool
		edit                func(c *EchoCanceller3Config)
		captureOutputUnused bool
	}
	variants := []variant{
		{name: "default", rate: 16000, numRender: 1, numCapture: 1},
		{name: "48k", rate: 48000, numRender: 1, numCapture: 2},
		{name: "short_initial_filter", rate: 16000, numRender: 1, numCapture: 1, edit: func(c *EchoCanceller3Config) {
			c.Filter.RefinedInitial.LengthBlocks = 9
			c.Filter.CoarseInitial.LengthBlocks = 9
			c.Filter.Refined.LengthBlocks = 20
			c.Filter.Coarse.LengthBlocks = 15
			c.Filter.ConfigChangeDurationBlocks = 40
			c.Filter.InitialStateSeconds = 0.5
		}},
		{name: "long_initial_filter", rate: 32000, numRender: 2, numCapture: 2, edit: func(c *EchoCanceller3Config) {
			c.Filter.RefinedInitial.LengthBlocks = 24
			c.Filter.Refined.LengthBlocks = 12
			c.Filter.InitialStateSeconds = 0.3
		}},
		{name: "linear_output_fixed_delay", rate: 16000, numRender: 1, numCapture: 1, edit: func(c *EchoCanceller3Config) {
			c.Filter.ExportLinearAecOutput = true
			c.Delay.FixedCaptureDelaySamples = 120
			c.Delay.NumFilters = 7
			c.Delay.DownSamplingFactor = 8
		}},
		{name: "erle_sections_subband_nearend", rate: 16000, numRender: 1, numCapture: 2, edit: func(c *EchoCanceller3Config) {
			c.Erle.NumSections = 3
			c.Suppressor.UseSubbandNearendDetection = true
			c.Suppressor.NearendAverageBlocks = 8
			c.Filter.HighPassFilterEchoReference = true
		}},
		{name: "stereo_content", rate: 48000, numRender: 2, numCapture: 2, multichannel: true, differentChannels: true},
		{name: "three_channels", rate: 16000, numRender: 3, numCapture: 3, differentChannels: true,
			edit: func(c *EchoCanceller3Config) { c.Filter.HighPassFilterEchoReference = true }},
		{name: "output_unused", rate: 16000, numRender: 1, numCapture: 1, captureOutputUnused: true},
	}
	for _, v := range variants {
		t.Run(v.name, func(t *testing.T) {
			cfg := DefaultEchoCanceller3Config()
			if v.edit != nil {
				v.edit(&cfg)
			}
			var multi *EchoCanceller3Config
			if v.multichannel {
				m := CreateDefaultMultichannelConfig()
				multi = &m
			}
			linearOutput := cfg.Filter.ExportLinearAecOutput
			once := NewEchoCanceller3(&cfg, multi, v.rate, v.numRender, v.numCapture)
			core := &Ec3Core{}
			core.Init(&cfg, multi, v.rate, v.numRender, v.numCapture, 0, nil)

			nb := NumBandsForRate(v.rate)
			rng := rand.New(rand.NewSource(int64(len(v.name)) + int64(v.rate)))
			noise := func(level float64) float32 { return float32((rng.Float64()*2 - 1) * level) }
			render := ec3MakeFrame(nb, v.numRender)
			captureA := ec3MakeFrame(nb, v.numCapture)
			captureB := ec3MakeFrame(nb, v.numCapture)
			var delayed [][]float32 // the lowest render band of the previous frames
			for f := 0; f < frames; f++ {
				level := 4000.0
				if f%150 >= 120 {
					level = 20
				}
				for b := range render {
					for k := range render[b][0] {
						x := noise(level)
						for ch := range render[b] {
							render[b][ch][k] = x
							// Stereo content in the middle third.
							if v.differentChannels && ch > 0 && f >= frames/3 && f < 2*frames/3 {
								render[b][ch][k] = noise(level)
							}
						}
					}
				}
				delayed = append(delayed, append([]float32(nil), render[0][0]...))
				for b := range captureA {
					for ch := range captureA[b] {
						for k := range captureA[b][ch] {
							var y float32
							if b == 0 && f >= 3 {
								y = float32(0.5-0.1*float64(ch)) * delayed[f-3][k]
							}
							if f >= frames/2 && f%7 < 3 {
								y += noise(600)
							}
							captureA[b][ch][k] = y
							captureB[b][ch][k] = y
						}
					}
				}
				levelChange := f == frames/4
				if v.captureOutputUnused && f == frames/2 {
					once.SetCaptureOutputUsage(false)
					var e EchoCanceller3
					core.Bind(&e)
					e.SetCaptureOutputUsage(false)
				}
				if f%50 == 10 {
					once.SetAudioBufferDelay(20 + f%3)
					var e EchoCanceller3
					core.Bind(&e)
					e.SetAudioBufferDelay(20 + f%3)
				}

				// Some frames come as a burst of two render frames.
				renders := 1
				if f%40 == 39 {
					renders = 2
				}
				for i := 0; i < renders; i++ {
					once.AnalyzeRender(render)
					var e EchoCanceller3
					core.BindRender(&e)
					e.AnalyzeRender(render)
				}

				var linearA, linearB [][]float32
				if linearOutput {
					linearA, linearB = make([][]float32, v.numCapture), make([][]float32, v.numCapture)
					for ch := range linearA {
						linearA[ch], linearB[ch] = make([]float32, SplitBandSize), make([]float32, SplitBandSize)
					}
				}
				once.ProcessCapture(captureA, linearA, levelChange)
				var e EchoCanceller3
				core.Bind(&e)
				e.ProcessCapture(captureB, linearB, levelChange)

				where := fmt.Sprintf("frame %d", f)
				for b := range captureA {
					for ch := range captureA[b] {
						for k := range captureA[b][ch] {
							if math.Float32bits(captureA[b][ch][k]) != math.Float32bits(captureB[b][ch][k]) {
								t.Fatalf("%s: band %d channel %d sample %d: %v with a view bound per call, %v with one view",
									where, b, ch, k, captureB[b][ch][k], captureA[b][ch][k])
							}
						}
					}
				}
				for ch := range linearA {
					for k := range linearA[ch] {
						if math.Float32bits(linearA[ch][k]) != math.Float32bits(linearB[ch][k]) {
							t.Fatalf("%s: linear output channel %d sample %d differs", where, ch, k)
						}
					}
				}
				var e2 EchoCanceller3
				core.Bind(&e2)
				if a, b := once.GetMetrics(), e2.GetMetrics(); a != b &&
					!(math.IsNaN(a.EchoReturnLoss) && math.IsNaN(b.EchoReturnLoss)) {
					t.Fatalf("%s: metrics %+v with a view bound per call, %+v with one view", where, b, a)
				}
				if once.StereoRenderProcessingActiveForTesting() != e2.StereoRenderProcessingActiveForTesting() {
					t.Fatalf("%s: stereo processing differs", where)
				}
			}
		})
	}
}
