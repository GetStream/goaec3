package goaec3

import (
	"errors"
	"math"
	"os"
	"strconv"
	"testing"
	"time"
)

// TestBadFrameShapes checks that malformed frames are rejected with an
// error and leave the canceller usable.
func TestBadFrameShapes(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		cfg := DefaultConfig()
		cfg.Filter.ExportLinearAecOutput = true
		e, err := New(cfg, rate, 2, 2)
		if err != nil {
			t.Fatal(err)
		}
		n := e.FrameSize()
		bad := map[string][][]float32{
			"nil":            nil,
			"empty":          {},
			"one channel":    robustFrame(1, n),
			"three channels": robustFrame(3, n),
			"zero length":    robustFrame(2, 0),
			"short":          robustFrame(2, n-1),
			"long":           robustFrame(2, n+1),
			"ragged":         {make([]float32, n), make([]float32, n-1)},
			"nil channel":    {make([]float32, n), nil},
			"double length":  robustFrame(2, 2*n),
		}
		for name, f := range bad {
			if err := e.AnalyzeRender(f); err == nil {
				t.Errorf("%d Hz AnalyzeRender %s: no error", rate, name)
			}
			if err := e.ProcessCapture(f); err == nil {
				t.Errorf("%d Hz ProcessCapture %s: no error", rate, name)
			}
			if err := e.ProcessCaptureWithLinearOutput(f, robustFrame(2, LinearOutputFrameSize)); err == nil {
				t.Errorf("%d Hz ProcessCaptureWithLinearOutput capture %s: no error", rate, name)
			}
		}
		for name, lin := range map[string][][]float32{
			"nil": nil, "one channel": robustFrame(1, LinearOutputFrameSize), "full rate": robustFrame(2, n+1),
			"short": robustFrame(2, LinearOutputFrameSize-1),
		} {
			if err := e.ProcessCaptureWithLinearOutput(robustFrame(2, n), lin); err == nil {
				t.Errorf("%d Hz linear output %s: no error", rate, name)
			}
		}
		// Still usable.
		scene := newRobustScene(1, rate, 20)
		render, capture := robustFrame(2, n), robustFrame(2, n)
		for range 100 {
			scene.render(render, robustSigEcho)
			scene.capture(capture, robustSigEcho)
			if err := e.AnalyzeRender(render); err != nil {
				t.Fatal(err)
			}
			if err := e.ProcessCapture(capture); err != nil {
				t.Fatal(err)
			}
			robustCheckFinite(t, "output", capture)
		}
	}
}

func TestNewRejectsBadArguments(t *testing.T) {
	for _, rate := range []int{-16000, 0, 8000, 22050, 44100, 96000, 16001} {
		if _, err := New(DefaultConfig(), rate, 1, 1); !errors.Is(err, ErrSampleRate) {
			t.Errorf("rate %d: %v", rate, err)
		}
	}
	for _, ch := range [][2]int{{0, 1}, {1, 0}, {-1, 1}, {1, -5}, {math.MinInt, 1}} {
		if _, err := New(DefaultConfig(), 16000, ch[0], ch[1]); !errors.Is(err, ErrNumChannels) {
			t.Errorf("channels %v: %v", ch, err)
		}
		if _, err := NewWithMultichannelConfig(DefaultConfig(), MultichannelConfig(), 16000, ch[0], ch[1]); !errors.Is(err,
			ErrNumChannels) {
			t.Errorf("multichannel channels %v: %v", ch, err)
		}
	}
}

// TestExtremeAudioBufferDelays sets audio buffer delays across the whole
// int range, in both delay modes, between and during render and capture.
func TestExtremeAudioBufferDelays(t *testing.T) {
	delays := []int{math.MinInt64, math.MinInt32, -1 << 20, -1000, -5, -1, 0, 1, 3, 4, 7, 100, 499, 1000, 5000,
		1 << 20, math.MaxInt32, math.MaxInt32 + 1, math.MaxInt64}
	for _, external := range []bool{false, true} {
		for _, rate := range []int{16000, 48000} {
			cfg := DefaultConfig()
			cfg.Delay.UseExternalDelayEstimator = external
			e, err := New(cfg, rate, 1, 1)
			if err != nil {
				t.Fatal(err)
			}
			n := e.FrameSize()
			scene := newRobustScene(3, rate, 50)
			render, capture := robustFrame(1, n), robustFrame(1, n)
			for i := range 30 * len(delays) {
				if i%15 == 0 {
					e.SetAudioBufferDelay(delays[(i/15)%len(delays)])
				}
				scene.render(render, robustSigEcho)
				scene.capture(capture, robustSigEcho)
				if err := e.AnalyzeRender(render); err != nil {
					t.Fatal(err)
				}
				if err := e.ProcessCapture(capture); err != nil {
					t.Fatal(err)
				}
				robustCheckFinite(t, "output", capture)
				m := e.Metrics()
				if math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement) {
					t.Fatalf("metrics %+v", m)
				}
			}
		}
	}
}

// TestChannelCounts runs 1 to 8 render and capture channels at each rate.
func TestChannelCounts(t *testing.T) {
	counts := [][2]int{{1, 1}, {1, 8}, {8, 1}, {8, 8}, {3, 5}}
	frames := 100
	if testing.Short() {
		frames = 30
	}
	for _, rate := range []int{16000, 32000, 48000} {
		for _, c := range counts {
			for _, mc := range []bool{false, true} {
				if mc && c[0] == 8 {
					continue
				}
				cfg := DefaultConfig()
				cfg.Filter.ExportLinearAecOutput = true
				var e *EchoCanceller
				var err error
				if mc {
					m := MultichannelConfig()
					robustAgree(&m, &cfg)
					e, err = NewWithMultichannelConfig(cfg, m, rate, c[0], c[1])
				} else {
					e, err = New(cfg, rate, c[0], c[1])
				}
				if err != nil {
					t.Fatal(err)
				}
				n := e.FrameSize()
				scene := newRobustScene(int64(c[0]), rate, 30)
				render, capture := robustFrame(c[0], n), robustFrame(c[1], n)
				linear := robustFrame(c[1], LinearOutputFrameSize)
				for range frames {
					scene.render(render, robustSigEcho)
					scene.capture(capture, robustSigEcho)
					if err := e.AnalyzeRender(render); err != nil {
						t.Fatal(err)
					}
					if err := e.ProcessCaptureWithLinearOutput(capture, linear); err != nil {
						t.Fatal(err)
					}
					robustCheckFinite(t, "output", capture)
					robustCheckFinite(t, "linear", linear)
				}
			}
		}
	}
}

// TestSpecialSampleValues feeds every special value class on each side
// after convergence; only NaN is excluded from the finiteness check (it
// propagates into the filter state, as in the C++).
func TestSpecialSampleValues(t *testing.T) {
	frames, rates := 400, []int{16000, 48000}
	if testing.Short() {
		frames, rates = 200, rates[:1]
	}
	for _, rate := range rates {
		for sig := 0; sig < robustNumSignals; sig++ {
			if sig == robustSigNaN {
				continue
			}
			e, err := New(DefaultConfig(), rate, 2, 1)
			if err != nil {
				t.Fatal(err)
			}
			n := e.FrameSize()
			scene := newRobustScene(5, rate, 40)
			render, capture := robustFrame(2, n), robustFrame(1, n)
			for i := range frames {
				rs, cs := robustSigEcho, robustSigEcho
				if i >= frames/2 {
					// Alternate the special value between the sides.
					if (i/20)%2 == 0 {
						rs = sig
					} else {
						cs = sig
					}
				}
				scene.render(render, rs)
				scene.capture(capture, cs)
				if err := e.AnalyzeRender(render); err != nil {
					t.Fatal(err)
				}
				if err := e.ProcessCapture(capture); err != nil {
					t.Fatal(err)
				}
				robustCheckFinite(t, "output", capture)
			}
		}
	}
}

// TestNaNInputDoesNotPanic feeds NaN samples on either side; the output may
// turn NaN (as libwebrtc's does), but nothing may panic, including on amd64
// where a NaN converted to an int is MinInt64.
func TestNaNInputDoesNotPanic(t *testing.T) {
	frames := 400
	if testing.Short() {
		frames = 150
	}
	for _, rate := range []int{16000, 32000, 48000} {
		for _, side := range []string{"render", "capture", "both"} {
			cfg := DefaultConfig()
			cfg.Filter.ExportLinearAecOutput = true
			e, err := New(cfg, rate, 1, 1)
			if err != nil {
				t.Fatal(err)
			}
			n := e.FrameSize()
			scene := newRobustScene(9, rate, 40)
			render, capture := robustFrame(1, n), robustFrame(1, n)
			linear := robustFrame(1, LinearOutputFrameSize)
			for i := range frames {
				rs, cs := robustSigEcho, robustSigEcho
				if i >= 100 && i < 110 {
					if side != "capture" {
						rs = robustSigNaN
					}
					if side != "render" {
						cs = robustSigNaN
					}
				}
				scene.render(render, rs)
				scene.capture(capture, cs)
				if err := e.AnalyzeRender(render); err != nil {
					t.Fatal(err)
				}
				if err := e.ProcessCaptureWithLinearOutput(capture, linear); err != nil {
					t.Fatal(err)
				}
				_ = e.Metrics()
			}
		}
	}
}

// robustSoakSeconds returns the length of the soak runs: short by default,
// GOAEC3_SOAK_SECONDS overrides it (3600 for the one-hour run).
func robustSoakSeconds(def int) int {
	if s := os.Getenv("GOAEC3_SOAK_SECONDS"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			return v
		}
	}
	if testing.Short() {
		return def / 8
	}
	return def
}

// TestOneSidedTraffic runs render-only and capture-only traffic for minutes
// of audio and then switches to normal traffic: the render transfer queue
// overflows and drops frames, the capture side runs without render, and the
// canceller must still converge afterwards.
func TestOneSidedTraffic(t *testing.T) {
	seconds := robustSoakSeconds(60)
	rates := []int{16000, 48000}
	if testing.Short() {
		rates = rates[:1]
	}
	for _, rate := range rates {
		for _, first := range []string{"render", "capture"} {
			for _, external := range []bool{false, true} {
				cfg := DefaultConfig()
				cfg.Delay.UseExternalDelayEstimator = external
				e, err := New(cfg, rate, 1, 1)
				if err != nil {
					t.Fatal(err)
				}
				if external {
					e.SetAudioBufferDelay(40)
				}
				n := e.FrameSize()
				scene := newRobustScene(11, rate, 40)
				render, capture := robustFrame(1, n), robustFrame(1, n)
				for range seconds * 100 {
					if first == "render" {
						scene.render(render, robustSigEcho)
						if err := e.AnalyzeRender(render); err != nil {
							t.Fatal(err)
						}
					} else {
						scene.capture(capture, robustSigEcho)
						if err := e.ProcessCapture(capture); err != nil {
							t.Fatal(err)
						}
						robustCheckFinite(t, "capture-only output", capture)
					}
				}
				var inPow, outPow float64
				for i := range 1000 {
					scene.render(render, robustSigEcho)
					scene.capture(capture, robustSigEcho)
					if i >= 500 {
						for _, v := range capture[0] {
							inPow += float64(v) * float64(v)
						}
					}
					if err := e.AnalyzeRender(render); err != nil {
						t.Fatal(err)
					}
					if err := e.ProcessCapture(capture); err != nil {
						t.Fatal(err)
					}
					robustCheckFinite(t, "output", capture)
					if i >= 500 {
						for _, v := range capture[0] {
							outPow += float64(v) * float64(v)
						}
					}
				}
				if !external {
					if r := 10 * math.Log10(inPow/outPow); r < 10 {
						t.Errorf("%d Hz, %s first: echo reduced by %.1f dB after one-sided traffic", rate, first, r)
					}
				}
			}
		}
	}
}

// TestLongRun processes a long stretch of audio (GOAEC3_SOAK_SECONDS, one
// hour for the full soak) and checks that the canceller keeps cancelling:
// the echo reduction over the last ten seconds stays above 20 dB and nothing
// turns non-finite. The delay estimate is logged, not checked: on this
// scene AEC3 (libwebrtc as well, bit-exactly) occasionally loses and
// re-acquires the delay during the tone segments.
func TestLongRun(t *testing.T) {
	seconds := robustSoakSeconds(40)
	for _, c := range []struct{ rate, numRender, numCapture int }{{16000, 1, 1}, {48000, 2, 2}} {
		start := time.Now()
		e, err := New(DefaultConfig(), c.rate, c.numRender, c.numCapture)
		if err != nil {
			t.Fatal(err)
		}
		n := e.FrameSize()
		scene := newRobustScene(13, c.rate, 60)
		render, capture := robustFrame(c.numRender, n), robustFrame(c.numCapture, n)
		frames := seconds * 100
		var inEarly, outEarly, inLate, outLate float64
		delay := -1
		for i := range frames {
			scene.render(render, robustSigEcho)
			scene.capture(capture, robustSigEcho)
			early := i >= 1000 && i < 2000
			late := i >= frames-1000
			if early || late {
				for _, v := range capture[0] {
					if early {
						inEarly += float64(v) * float64(v)
					} else {
						inLate += float64(v) * float64(v)
					}
				}
			}
			if err := e.AnalyzeRender(render); err != nil {
				t.Fatal(err)
			}
			if err := e.ProcessCapture(capture); err != nil {
				t.Fatal(err)
			}
			robustCheckFinite(t, "output", capture)
			if early || late {
				for _, v := range capture[0] {
					if early {
						outEarly += float64(v) * float64(v)
					} else {
						outLate += float64(v) * float64(v)
					}
				}
			}
			if i == 2000 {
				delay = e.Metrics().DelayMs
			}
		}
		m := e.Metrics()
		if math.IsNaN(m.EchoReturnLoss) || math.IsInf(m.EchoReturnLoss, 0) ||
			math.IsNaN(m.EchoReturnLossEnhancement) || math.IsInf(m.EchoReturnLossEnhancement, 0) {
			t.Errorf("%+v: metrics %+v", c, m)
		}
		if frames > 3000 {
			early := 10 * math.Log10(inEarly/outEarly)
			late := 10 * math.Log10(inLate/outLate)
			if late < 20 {
				t.Errorf("%+v: echo reduction %.1f dB after convergence, %.1f dB at the end", c, early, late)
			}
			t.Logf("%+v: %d s of audio in %v; echo reduction %.1f dB -> %.1f dB, delay %d ms at 20 s, metrics %+v",
				c, seconds, time.Since(start).Round(time.Millisecond), early, late, delay, m)
		}
	}
}

// TestSteadyStateAllocations locks the steady state of several setups to
// zero allocations per frame, Metrics included.
func TestSteadyStateAllocations(t *testing.T) {
	warmup, runs := 400, 200
	if testing.Short() {
		warmup, runs = 100, 20
	}
	cases := []struct {
		name                        string
		rate, numRender, numCapture int
		mod                         func(*Config)
		bufferDelayMs               int
	}{
		{"16k_mono", 16000, 1, 1, func(*Config) {}, 0},
		{"32k_linear", 32000, 1, 2, func(c *Config) { c.Filter.ExportLinearAecOutput = true }, 0},
		{"48k_stereo_no_detection", 48000, 2, 2, func(c *Config) { c.MultiChannel.DetectStereoContent = false }, 0},
		{"48k_external", 48000, 1, 1, func(c *Config) { c.Delay.UseExternalDelayEstimator = true }, 50},
		{"16k_subband_hpf_fixed_delay", 16000, 1, 1, func(c *Config) {
			c.Suppressor.UseSubbandNearendDetection = true
			c.Filter.HighPassFilterEchoReference = true
			c.Delay.FixedCaptureDelaySamples = 100
		}, 0},
	}
	for _, c := range cases {
		cfg := DefaultConfig()
		c.mod(&cfg)
		e, err := New(cfg, c.rate, c.numRender, c.numCapture)
		if err != nil {
			t.Fatal(err)
		}
		if c.bufferDelayMs > 0 {
			e.SetAudioBufferDelay(c.bufferDelayMs)
		}
		n := e.FrameSize()
		scene := newRobustScene(17, c.rate, 30)
		render, capture := robustFrame(c.numRender, n), robustFrame(c.numCapture, n)
		linear := robustFrame(c.numCapture, LinearOutputFrameSize)
		step := func() {
			scene.render(render, robustSigEcho)
			scene.capture(capture, robustSigEcho)
			_ = e.AnalyzeRender(render)
			if cfg.Filter.ExportLinearAecOutput {
				_ = e.ProcessCaptureWithLinearOutput(capture, linear)
			} else {
				_ = e.ProcessCapture(capture)
			}
			_ = e.Metrics()
		}
		for range warmup {
			step()
		}
		if a := testing.AllocsPerRun(runs, step); a != 0 {
			t.Errorf("%s: %v allocations per frame", c.name, a)
		}
	}
}

// TestBoundaryConfigSweep sets each config field, one at a time, to the
// extremes of its type, its edge values and the Validate bounds, validates
// the config and runs echo audio through three base setups (16 kHz mono;
// 48 kHz stereo with subband nearend detection and linear output; 32 kHz
// external delay). Nothing may panic and, for finite fields, the output must
// stay finite, and the configs New rejects (robustRejected) must return
// ErrConfig.
func TestBoundaryConfigSweep(t *testing.T) {
	frames := 40
	if testing.Short() {
		frames = 6
	}
	failures := robustSweepFailures(t, frames, func(path string, cfg *Config) bool {
		if robustRejected(cfg) {
			// The documented ErrConfig set: New must refuse it.
			if _, err := New(*cfg, 16000, 1, 1); !errors.Is(err, ErrConfig) {
				t.Errorf("%s: New error = %v, want ErrConfig", path, err)
			}
			return true
		}
		return !robustRenderPowerFinite(cfg)
	})
	for _, f := range failures {
		t.Error(f)
	}
}
