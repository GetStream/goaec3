package goaec3

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// Open robustness findings. Each test demonstrates one finding and skips
// until the fix lands; robustPendingFix keeps FuzzEchoCanceller and the
// boundary sweep away from the same inputs meanwhile. Remove the skip and
// the matching robustPendingFix clause together.

// robustSkipKnown skips a known-failure test unless
// GOAEC3_RUN_KNOWN_FAILURES is set.
func robustSkipKnown(t *testing.T, reason string) {
	t.Helper()
	if os.Getenv("GOAEC3_RUN_KNOWN_FAILURES") == "" {
		t.Skip(reason)
	}
}

// robustPendingFix reports whether cfg hits an open finding.
func robustPendingFix(cfg *Config) bool {
	sn := &cfg.Suppressor.SubbandNearendDetection
	return cfg.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold == 0 ||
		cfg.Delay.NumFilters == 0 ||
		cfg.Filter.Refined.LengthBlocks < 9 ||
		(cfg.Suppressor.UseSubbandNearendDetection && (sn.Subband1.High > 64 || sn.Subband2.High > 64))
}

// robustRunNoPanic creates a canceller and runs frames echo frames; it
// returns the panic message, or "" when New returns an error or the run
// completes.
func robustRunNoPanic(newEC func() (*EchoCanceller, error), frames int) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprint(r)
		}
	}()
	e, err := newEC()
	if err != nil {
		return ""
	}
	scene := newRobustScene(1, e.SampleRateHz(), 40)
	render := robustFrame(e.NumRenderChannels(), e.FrameSize())
	capture := robustFrame(e.NumCaptureChannels(), e.FrameSize())
	for range frames {
		scene.render(render, robustSigEcho)
		if err := e.AnalyzeRender(render); err != nil {
			return err.Error()
		}
		scene.capture(capture, robustSigEcho)
		if err := e.ProcessCapture(capture); err != nil {
			return err.Error()
		}
		for _, x := range capture {
			for _, v := range x {
				if v != v {
					return "NaN output for finite input"
				}
			}
		}
	}
	return ""
}

// TestKnownFailureAntiHowlingNaN: with
// Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0 (the
// lower Validate bound), SuppressionGain::UpperBandsGain computes
// anti_howling_gain = sqrtf(0 / 0) = NaN while the band energies are zero.
// The C++ returns std::min(std::min(gain_below_8_khz, anti_howling_gain),
// gain_bound); std::min(a, NaN) is a, so the gain stays finite. The port
// uses the builtin min, which returns NaN when any argument is NaN, and the
// output turns NaN (at 48 kHz within the first frames). The differential
// fuzz against libwebrtc reproduces it as a bit-exactness failure.
func TestKnownFailureAntiHowlingNaN(t *testing.T) {
	robustSkipKnown(t, "open finding: builtin min vs std::min on NaN in SuppressionGain.upperBandsGain")
	for _, rate := range []int{48000} {
		cfg := DefaultConfig()
		cfg.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0
		if msg := robustRunNoPanic(func() (*EchoCanceller, error) { return New(cfg, rate, 1, 1) }, 200); msg != "" {
			t.Errorf("%d Hz: %s", rate, msg)
		}
	}
}

// TestKnownFailureConfigsTheCppCannotRun: Validate accepts these configs,
// but the C++ cannot run them (the EchoCanceller3 constructor segfaults with
// zero matched filters and aborts on a negative vector size with a refined
// filter shorter than 9 blocks; a subband nearend region ending at bin 65
// reads past the 65-bin spectrum), and the port panics. New should reject
// them with an error instead (the planned ErrConfig; once it exists, assert
// errors.Is(err, ErrConfig) here). TestConfigsNextToTheCrashSetRun pins the
// neighboring values that must stay accepted.
func TestKnownFailureConfigsTheCppCannotRun(t *testing.T) {
	robustSkipKnown(t, "open finding: Validate-legal configs the C++ cannot run make the port panic")
	cases := map[string]func(*Config){
		"NumFilters=0":    func(c *Config) { c.Delay.NumFilters = 0 },
		"RefinedLength=1": func(c *Config) { c.Filter.Refined.LengthBlocks = 1 },
		"RefinedLength=8": func(c *Config) { c.Filter.Refined.LengthBlocks = 8 },
		"Subband1High=65": func(c *Config) {
			c.Suppressor.UseSubbandNearendDetection = true
			c.Suppressor.SubbandNearendDetection.Subband1.High = 65
		},
		"Subband2Low=65": func(c *Config) {
			c.Suppressor.UseSubbandNearendDetection = true
			c.Suppressor.SubbandNearendDetection.Subband2.Low = 65
			c.Suppressor.SubbandNearendDetection.Subband2.High = 65
		},
	}
	for name, mod := range cases {
		cfg := DefaultConfig()
		mod(&cfg)
		cfg.Validate()
		for _, mc := range []bool{false, true} {
			newEC := func() (*EchoCanceller, error) { return New(cfg, 16000, 1, 1) }
			if mc {
				// The same value in the multichannel config only, active
				// from the start.
				mono, multi := DefaultConfig(), cfg
				mono.MultiChannel.DetectStereoContent = false
				multi.MultiChannel.DetectStereoContent = false
				newEC = func() (*EchoCanceller, error) {
					return NewWithMultichannelConfig(mono, multi, 16000, 2, 1)
				}
			}
			if msg := robustRunNoPanic(newEC, 50); msg != "" {
				t.Errorf("%s (multichannel config %v): %s", name, mc, msg)
			}
		}
	}
}

// TestKnownFailureMultichannelConfigMismatch: NewWithMultichannelConfig
// documents that the two configs must agree on the linear output and the
// fixed capture delay, but does not check it. When the multichannel config
// becomes active (from the start with more than one render channel and
// stereo detection off, or once stereo content is detected) and disagrees,
// ProcessCapture panics; the C++ dereferences a null linear output framer or
// block delay buffer. NewWithMultichannelConfig should return an error when
// the configs fail the C++ CompatibleConfigs check (config_selector.cc):
// equal Delay.FixedCaptureDelaySamples, Filter.ExportLinearAecOutput,
// Filter.HighPassFilterEchoReference, MultiChannel.DetectStereoContent and
// MultiChannel.StereoDetectionTimeoutThresholdSeconds.
func TestKnownFailureMultichannelConfigMismatch(t *testing.T) {
	robustSkipKnown(t, "open finding: NewWithMultichannelConfig accepts configs that disagree")
	cases := []struct {
		name   string
		detect bool
		frames int
		mod    func(cfg, mc *Config)
	}{
		{"linear output, multichannel from the start", false, 50, func(cfg, mc *Config) {
			cfg.Filter.ExportLinearAecOutput = true
			mc.Filter.ExportLinearAecOutput = false
		}},
		{"fixed capture delay, after stereo detection", true, 600, func(cfg, mc *Config) {
			cfg.Delay.FixedCaptureDelaySamples = 0
			mc.Delay.FixedCaptureDelaySamples = 100
		}},
	}
	for _, c := range cases {
		cfg, mc := DefaultConfig(), MultichannelConfig()
		cfg.MultiChannel.DetectStereoContent = c.detect
		mc.MultiChannel.DetectStereoContent = c.detect
		c.mod(&cfg, &mc)
		newEC := func() (*EchoCanceller, error) { return NewWithMultichannelConfig(cfg, mc, 16000, 2, 1) }
		if msg := robustRunNoPanic(newEC, c.frames); msg != "" {
			t.Errorf("%s: %s", c.name, msg)
		}
	}
}

// TestKnownFailureNotifyEchoPathGainChangeRace: the EchoCanceller doc lists
// the capture-side methods that must be serialized with ProcessCapture but
// not NotifyEchoPathGainChange, which writes a plain bool that
// processCapture reads and clears. Called from another goroutine (for
// instance the one that changes the analog gain) it races with
// ProcessCapture; go test -race reports it. Either the doc names it as a
// capture-side method, or the flag becomes an atomic.Bool (read and cleared
// with Swap(false) in processCapture).
func TestKnownFailureNotifyEchoPathGainChangeRace(t *testing.T) {
	robustSkipKnown(t, "open finding: NotifyEchoPathGainChange is not synchronized with ProcessCapture")
	e, err := New(DefaultConfig(), 16000, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 200 {
			e.NotifyEchoPathGainChange()
		}
	}()
	scene := newRobustScene(1, 16000, 20)
	render := robustFrame(1, 160)
	capture := robustFrame(1, 160)
	for range 200 {
		scene.render(render, robustSigEcho)
		scene.capture(capture, robustSigEcho)
		_ = e.AnalyzeRender(render)
		_ = e.ProcessCapture(capture)
	}
	<-done
}

// TestConfigsNextToTheCrashSetRun runs the values next to the crash set of
// TestKnownFailureConfigsTheCppCannotRun, which libwebrtc runs and New must
// keep accepting.
func TestConfigsNextToTheCrashSetRun(t *testing.T) {
	cases := map[string]func(*Config){
		"NumFilters=1":    func(c *Config) { c.Delay.NumFilters = 1 },
		"RefinedLength=9": func(c *Config) { c.Filter.Refined.LengthBlocks = 9 },
		"Subband1High=64": func(c *Config) {
			c.Suppressor.UseSubbandNearendDetection = true
			c.Suppressor.SubbandNearendDetection.Subband1.High = 64
		},
		"Subband2Low=64": func(c *Config) {
			c.Suppressor.UseSubbandNearendDetection = true
			c.Suppressor.SubbandNearendDetection.Subband2.Low = 64
			c.Suppressor.SubbandNearendDetection.Subband2.High = 64
		},
		"AntiHowlingActivationThreshold=1e-6": func(c *Config) {
			c.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 1e-6
		},
	}
	for name, mod := range cases {
		cfg := DefaultConfig()
		mod(&cfg)
		cfg.Validate()
		for _, rate := range []int{16000, 48000} {
			e, err := New(cfg, rate, 1, 1)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if msg := robustRunNoPanic(func() (*EchoCanceller, error) { return e, nil }, 100); msg != "" {
				t.Errorf("%s at %d Hz: %s", name, rate, msg)
			}
		}
	}
}

// TestKnownFailureUnvalidatedConfigs: New documents that the config is used
// as given and that Validate should be called first, but it does not check.
// Without Validate these single-field configs panic in New or in the first
// frames (makeslice with a negative length, integer divide by zero,
// negative indices, "Decimator requires a down-sampling factor of 4 or 8").
// Suppressor.NearendAverageBlocks = MaxInt32 (also rejected by Validate)
// makes New allocate until the process is killed, which recover cannot
// catch, so it is not run here. New should validate a copy of each config
// and return ErrConfig when Validate reports false.
func TestKnownFailureUnvalidatedConfigs(t *testing.T) {
	robustSkipKnown(t, "open finding: New does not reject configs that fail Validate")
	cases := map[string]func(*Config){
		"DownSamplingFactor=3":               func(c *Config) { c.Delay.DownSamplingFactor = 3 },
		"DownSamplingFactor=0":               func(c *Config) { c.Delay.DownSamplingFactor = 0 },
		"DownSamplingFactor=100":             func(c *Config) { c.Delay.DownSamplingFactor = 100 },
		"NumFilters=-2":                      func(c *Config) { c.Delay.NumFilters = -2 },
		"DelayHeadroomSamples=MinInt32":      func(c *Config) { c.Delay.DelayHeadroomSamples = math.MinInt32 },
		"Refined.LengthBlocks=-2":            func(c *Config) { c.Filter.Refined.LengthBlocks = -2 },
		"Coarse.LengthBlocks=-2":             func(c *Config) { c.Filter.Coarse.LengthBlocks = -2 },
		"RefinedInitial.LengthBlocks=-2":     func(c *Config) { c.Filter.RefinedInitial.LengthBlocks = -2 },
		"CoarseInitial.LengthBlocks=-2":      func(c *Config) { c.Filter.CoarseInitial.LengthBlocks = -2 },
		"Suppressor.NearendAverageBlocks=-2": func(c *Config) { c.Suppressor.NearendAverageBlocks = -2 },
	}
	for name, mod := range cases {
		cfg := DefaultConfig()
		mod(&cfg)
		if msg := robustRunNoPanic(func() (*EchoCanceller, error) { return New(cfg, 16000, 1, 1) }, 50); msg != "" {
			t.Errorf("%s: %s", name, msg)
		}
	}
}
