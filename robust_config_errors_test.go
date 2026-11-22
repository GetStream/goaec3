package goaec3

import (
	"errors"
	"math"
	"testing"
)

// TestConfigsTheCppCannotRunAreRejected: Validate accepts these configs, but
// libwebrtc cannot run them (the EchoCanceller3 constructor segfaults with
// zero matched filters and aborts on a negative vector size with a refined
// filter shorter than 9 blocks; a subband nearend region ending at bin 65
// reads past the 65-bin spectrum; a delay headroom of the initial refined
// filter length or more indexes past the frequency responses of the filter; a refined filter ErrorFloor or NoiseGate
// of 0 turns the output into NaN for good once the capture signal is digital
// silence). New rejects them with ErrConfig, also
// when only the multichannel config carries the value.
// TestConfigsNextToTheCrashSetRun pins the neighboring values that must stay
// accepted.
func TestConfigsTheCppCannotRunAreRejected(t *testing.T) {
	cases := map[string]func(*Config){
		"NumFilters=0":                    func(c *Config) { c.Delay.NumFilters = 0 },
		"RefinedLength=1":                 func(c *Config) { c.Filter.Refined.LengthBlocks = 1 },
		"RefinedLength=8":                 func(c *Config) { c.Filter.Refined.LengthBlocks = 8 },
		"Refined.ErrorFloor=0":            func(c *Config) { c.Filter.Refined.ErrorFloor = 0 },
		"RefinedInitial.ErrorFloor=0":     func(c *Config) { c.Filter.RefinedInitial.ErrorFloor = 0 },
		"RefinedInitial.ErrorFloor=1e-45": func(c *Config) { c.Filter.RefinedInitial.ErrorFloor = 1e-45 },
		"Refined.ErrorCeil=0":             func(c *Config) { c.Filter.Refined.ErrorCeil = 0 },
		"RefinedInitial.ErrorCeil=1e-45":  func(c *Config) { c.Filter.RefinedInitial.ErrorCeil = 1e-45 },
		"Refined.NoiseGate=0":             func(c *Config) { c.Filter.Refined.NoiseGate = 0 },
		"RefinedInitial.NoiseGate=0":      func(c *Config) { c.Filter.RefinedInitial.NoiseGate = 0 },
		"RefinedInitial.NoiseGate=9e-13":  func(c *Config) { c.Filter.RefinedInitial.NoiseGate = 9e-13 },
		"DelayHeadroomSamples=768":        func(c *Config) { c.Delay.DelayHeadroomSamples = 768 },
		"DelayHeadroomSamples=1000":       func(c *Config) { c.Delay.DelayHeadroomSamples = 1000 },
		"DelayHeadroomSamples=5000":       func(c *Config) { c.Delay.DelayHeadroomSamples = 5000 },
		"DelayHeadroomSamples=64, RefinedInitial.LengthBlocks=1": func(c *Config) {
			c.Delay.DelayHeadroomSamples = 64
			c.Filter.RefinedInitial.LengthBlocks = 1
		},
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
		if _, err := New(cfg, 16000, 1, 1); !errors.Is(err, ErrConfig) {
			t.Errorf("%s: New error = %v, want ErrConfig", name, err)
		}
		mono, multi := DefaultConfig(), cfg
		mono.MultiChannel.DetectStereoContent = false
		multi.MultiChannel.DetectStereoContent = false
		if _, err := NewWithMultichannelConfig(mono, multi, 16000, 2, 1); !errors.Is(err, ErrConfig) {
			t.Errorf("%s (multichannel config): error = %v, want ErrConfig", name, err)
		}
	}
}

// TestUnvalidatedConfigsAreRejected: configs that Config.Validate would
// change panic or exhaust memory in the pipeline (makeslice with a negative
// length, integer divide by zero, negative indices, a decimator factor other
// than 4 or 8, Suppressor.NearendAverageBlocks = MaxInt32). New rejects them
// with ErrConfig before allocating anything.
func TestUnvalidatedConfigsAreRejected(t *testing.T) {
	cases := map[string]func(*Config){
		"DownSamplingFactor=3":                     func(c *Config) { c.Delay.DownSamplingFactor = 3 },
		"DownSamplingFactor=0":                     func(c *Config) { c.Delay.DownSamplingFactor = 0 },
		"DownSamplingFactor=100":                   func(c *Config) { c.Delay.DownSamplingFactor = 100 },
		"NumFilters=-2":                            func(c *Config) { c.Delay.NumFilters = -2 },
		"DelayHeadroomSamples=MinInt32":            func(c *Config) { c.Delay.DelayHeadroomSamples = math.MinInt32 },
		"Refined.LengthBlocks=-2":                  func(c *Config) { c.Filter.Refined.LengthBlocks = -2 },
		"Coarse.LengthBlocks=-2":                   func(c *Config) { c.Filter.Coarse.LengthBlocks = -2 },
		"RefinedInitial.LengthBlocks=-2":           func(c *Config) { c.Filter.RefinedInitial.LengthBlocks = -2 },
		"CoarseInitial.LengthBlocks=-2":            func(c *Config) { c.Filter.CoarseInitial.LengthBlocks = -2 },
		"Suppressor.NearendAverageBlocks=-2":       func(c *Config) { c.Suppressor.NearendAverageBlocks = -2 },
		"Suppressor.NearendAverageBlocks=MaxInt32": func(c *Config) { c.Suppressor.NearendAverageBlocks = math.MaxInt32 },
		"Buffering.MaxAllowedExcessRenderBlocks=-1": func(c *Config) {
			c.Buffering.MaxAllowedExcessRenderBlocks = -1
		},
		"Buffering.ExcessRenderDetectionIntervalBlocks=-1": func(c *Config) {
			c.Buffering.ExcessRenderDetectionIntervalBlocks = -1
		},
	}
	for name, mod := range cases {
		cfg := DefaultConfig()
		mod(&cfg)
		if _, err := New(cfg, 16000, 1, 1); !errors.Is(err, ErrConfig) {
			t.Errorf("%s: New error = %v, want ErrConfig", name, err)
		}
	}
}

// TestMultichannelConfigMismatchIsRejected: the configs passed to
// NewWithMultichannelConfig must pass the C++ CompatibleConfigs check
// (config_selector.cc); a mismatch makes the C++ dereference a null linear
// output framer or block delay buffer once the multichannel config becomes
// active.
func TestMultichannelConfigMismatchIsRejected(t *testing.T) {
	for name, mod := range map[string]func(cfg, mc *Config){
		"ExportLinearAecOutput":       func(cfg, mc *Config) { cfg.Filter.ExportLinearAecOutput = true },
		"FixedCaptureDelaySamples":    func(cfg, mc *Config) { mc.Delay.FixedCaptureDelaySamples = 100 },
		"HighPassFilterEchoReference": func(cfg, mc *Config) { mc.Filter.HighPassFilterEchoReference = true },
		"DetectStereoContent":         func(cfg, mc *Config) { mc.MultiChannel.DetectStereoContent = false },
		"StereoDetectionTimeout": func(cfg, mc *Config) {
			mc.MultiChannel.StereoDetectionTimeoutThresholdSeconds = 10
		},
	} {
		cfg, mc := DefaultConfig(), MultichannelConfig()
		mod(&cfg, &mc)
		if _, err := NewWithMultichannelConfig(cfg, mc, 16000, 2, 1); !errors.Is(err, ErrConfig) {
			t.Errorf("%s: error = %v, want ErrConfig", name, err)
		}
	}
	if _, err := NewWithMultichannelConfig(DefaultConfig(), MultichannelConfig(), 16000, 2, 1); err != nil {
		t.Fatalf("default configs: %v", err)
	}
}

// TestNotifyEchoPathGainChangeConcurrent calls NotifyEchoPathGainChange from
// another goroutine while capture runs; go test -race checks it.
func TestNotifyEchoPathGainChangeConcurrent(t *testing.T) {
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

// robustRejected is the documented set of validated configs New refuses:
// configs libwebrtc cannot run, negative Buffering fields (size_t in C++,
// untouched by Validate) and configs over the resource limits.
func robustRejected(cfg *Config) bool {
	sn := &cfg.Suppressor.SubbandNearendDetection
	return cfg.Delay.NumFilters == 0 ||
		cfg.Filter.Refined.LengthBlocks < 9 ||
		(cfg.Suppressor.UseSubbandNearendDetection && (sn.Subband1.High > 64 || sn.Subband2.High > 64)) ||
		cfg.Buffering.ExcessRenderDetectionIntervalBlocks < 0 ||
		cfg.Buffering.MaxAllowedExcessRenderBlocks < 0 ||
		cfg.Delay.DelayHeadroomSamples/64 >= cfg.Filter.RefinedInitial.LengthBlocks ||
		ec3RefinedStepUndefined(&cfg.Filter.Refined) || ec3RefinedStepUndefined(&cfg.Filter.RefinedInitial) ||
		cfg.Filter.Refined.LengthBlocks > MaxFilterLengthBlocks ||
		cfg.Filter.Coarse.LengthBlocks > MaxFilterLengthBlocks ||
		cfg.Delay.NumFilters > MaxDelayFilters
}
