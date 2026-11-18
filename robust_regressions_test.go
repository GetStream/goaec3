package goaec3

import (
	"fmt"
	"testing"
)

// Regression tests for robustness findings that are fixed.

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

// TestAntiHowlingThresholdZeroStaysFinite: with
// Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0 (the
// lower Validate bound), SuppressionGain::UpperBandsGain computes
// anti_howling_gain = sqrtf(0 / 0) = NaN while the band energies are zero.
// The C++ returns std::min(std::min(gain_below_8_khz, anti_howling_gain),
// gain_bound); std::min(a, NaN) is a, so the gain stays finite, and so does
// the port, which follows std::min/std::max semantics.
func TestAntiHowlingThresholdZeroStaysFinite(t *testing.T) {
	for _, rate := range []int{16000, 32000, 48000} {
		cfg := DefaultConfig()
		cfg.Suppressor.HighBandsSuppression.AntiHowlingActivationThreshold = 0
		if msg := robustRunNoPanic(func() (*EchoCanceller, error) { return New(cfg, rate, 1, 1) }, 200); msg != "" {
			t.Errorf("%d Hz: %s", rate, msg)
		}
	}
}

// TestConfigsNextToTheCrashSetRun runs the values next to the crash set of
// TestConfigsTheCppCannotRunAreRejected, which libwebrtc runs and New must
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
