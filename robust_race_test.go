package goaec3

import (
	"math"
	"sync"
	"testing"
)

// TestConcurrentRenderAndCapture exercises the documented threading
// contract: AnalyzeRender on one goroutine, the capture-side methods on
// another (as webrtc::EchoCanceller3 is driven from the render and capture
// threads through its render transfer queue). Run with -race. The output
// depends on the interleaving, so only freedom from races and panics and a
// finite output are required. The stereo case switches to multichannel
// processing mid-run, which re-creates the block processor on the capture
// side while render frames keep arriving.
func TestConcurrentRenderAndCapture(t *testing.T) {
	cases := []struct {
		name                        string
		rate, numRender, numCapture int
		cfg                         func() Config
		mc                          bool
	}{
		{"16k_mono", 16000, 1, 1, DefaultConfig, false},
		{"48k_stereo_multichannel", 48000, 2, 2, func() Config {
			c := DefaultConfig()
			c.Filter.ExportLinearAecOutput = true
			return c
		}, true},
		{"32k_external_delay", 32000, 1, 1, func() Config {
			c := DefaultConfig()
			c.Delay.UseExternalDelayEstimator = true
			return c
		}, false},
	}
	frames := 600
	if testing.Short() {
		frames = 150
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := c.cfg()
			var e *EchoCanceller
			var err error
			if c.mc {
				mc := MultichannelConfig()
				robustAgree(&mc, &cfg)
				e, err = NewWithMultichannelConfig(cfg, mc, c.rate, c.numRender, c.numCapture)
			} else {
				e, err = New(cfg, c.rate, c.numRender, c.numCapture)
			}
			if err != nil {
				t.Fatal(err)
			}
			n := e.FrameSize()
			renderScene := newRobustScene(1, c.rate, 30)
			captureScene := newRobustScene(1, c.rate, 30)
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				render := robustFrame(c.numRender, n)
				for range frames {
					renderScene.render(render, robustSigEcho)
					if err := e.AnalyzeRender(render); err != nil {
						t.Error(err)
						return
					}
				}
			}()
			capture := robustFrame(c.numCapture, n)
			linear := robustFrame(c.numCapture, LinearOutputFrameSize)
			stereoSeen := false
			for i := range frames {
				// The capture scene replays the render signal on this
				// goroutine to build an echo.
				captureScene.render(robustFrame(1, n), robustSigEcho)
				captureScene.capture(capture, robustSigEcho)
				switch i % 50 {
				case 10:
					e.SetAudioBufferDelay(40)
				case 20:
					e.SetCaptureOutputUsage(i%100 == 20)
				case 30:
					_ = e.Metrics()
				}
				if cfg.Filter.ExportLinearAecOutput {
					err = e.ProcessCaptureWithLinearOutput(capture, linear)
				} else {
					err = e.ProcessCapture(capture)
				}
				if err != nil {
					t.Fatal(err)
				}
				robustCheckFinite(t, "capture output", capture)
				m := e.Metrics()
				if math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement) {
					t.Fatalf("metrics %+v", m)
				}
				stereoSeen = stereoSeen || e.ec.StereoRenderProcessingActiveForTesting()
			}
			wg.Wait()
			// Stereo processing may end again when the render goroutine
			// finishes early, so the test asks whether it was ever active.
			if c.mc && !testing.Short() && !stereoSeen {
				t.Error("stereo processing never became active: the re-initialization was not exercised")
			}
		})
	}
}

// TestConcurrentEchoCancellers runs independent cancellers in parallel:
// they share no mutable state (FFT tables and other package-level data are
// read-only).
func TestConcurrentEchoCancellers(t *testing.T) {
	var wg sync.WaitGroup
	outputs := make([][]float32, 4)
	for i := range outputs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			e, err := New(DefaultConfig(), 16000, 1, 1)
			if err != nil {
				t.Error(err)
				return
			}
			scene := newRobustScene(7, 16000, 20)
			render := robustFrame(1, 160)
			capture := robustFrame(1, 160)
			for range 300 {
				scene.render(render, robustSigEcho)
				scene.capture(capture, robustSigEcho)
				_ = e.AnalyzeRender(render)
				_ = e.ProcessCapture(capture)
				outputs[i] = append(outputs[i], capture[0]...)
			}
		}()
	}
	wg.Wait()
	for i := 1; i < len(outputs); i++ {
		for k := range outputs[0] {
			if math.Float32bits(outputs[i][k]) != math.Float32bits(outputs[0][k]) {
				t.Fatalf("canceller %d sample %d: %v, canceller 0: %v", i, k, outputs[i][k], outputs[0][k])
			}
		}
	}
}
