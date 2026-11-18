package goaec3

import (
	"fmt"
	"math"
	"reflect"
	"runtime/debug"
	"strings"
	"testing"
)

// robustSweepBase is a starting configuration for the boundary sweep.
type robustSweepBase struct {
	name                        string
	rate, numRender, numCapture int
	setup                       func(cfg *Config)
	bufferDelayMs               int // SetAudioBufferDelay before the run when > 0
}

var robustSweepBases = []robustSweepBase{
	{name: "default_16k_1x1", rate: 16000, numRender: 1, numCapture: 1, setup: func(*Config) {}},
	{name: "subband_linear_48k_2x2", rate: 48000, numRender: 2, numCapture: 2, setup: func(c *Config) {
		c.Suppressor.UseSubbandNearendDetection = true
		c.Filter.ExportLinearAecOutput = true
	}},
	{name: "external_32k_1x2", rate: 32000, numRender: 1, numCapture: 2, bufferDelayMs: 60, setup: func(c *Config) {
		c.Delay.UseExternalDelayEstimator = true
	}},
}

// robustSweepRun runs frames render/capture pairs of the echo scene and
// returns the panic, if any, with its stack.
func robustSweepRun(cfg Config, b robustSweepBase, frames int) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg = fmt.Sprintf("%v\n%s", r, debug.Stack())
		}
	}()
	e, err := New(cfg, b.rate, b.numRender, b.numCapture)
	if err != nil {
		return err.Error()
	}
	if b.bufferDelayMs > 0 {
		e.SetAudioBufferDelay(b.bufferDelayMs)
	}
	scene := newRobustScene(1, b.rate, 40)
	render := robustFrame(b.numRender, e.FrameSize())
	capture := robustFrame(b.numCapture, e.FrameSize())
	linear := robustFrame(b.numCapture, LinearOutputFrameSize)
	for range frames {
		scene.render(render, robustSigEcho)
		if err := e.AnalyzeRender(render); err != nil {
			return err.Error()
		}
		scene.capture(capture, robustSigEcho)
		if cfg.Filter.ExportLinearAecOutput {
			err = e.ProcessCaptureWithLinearOutput(capture, linear)
		} else {
			err = e.ProcessCapture(capture)
		}
		if err != nil {
			return err.Error()
		}
		for _, x := range capture {
			for _, v := range x {
				if v != v || v > 1 || v < -1 {
					return fmt.Sprintf("output sample %v", v)
				}
			}
		}
		if m := e.Metrics(); math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement) {
			return fmt.Sprintf("metrics %+v", m)
		}
	}
	return ""
}

// robustSweepValues returns the values the sweep sets a leaf to before
// Validate: the extremes of its type (which Validate clamps to the bounds
// of validated fields) and the small edge values of unvalidated ones.
func robustSweepValues(kind reflect.Kind) []any {
	switch kind {
	case reflect.Bool:
		return []any{false, true}
	case reflect.Int:
		return []any{math.MinInt32, -1, 0, 1, math.MaxInt32}
	default:
		return []any{float32(-math.MaxFloat32), float32(-1), float32(0), float32(1), float32(math.MaxFloat32),
			float32(math.NaN()), float32(math.Inf(1))}
	}
}

// robustSweepFailures runs the boundary sweep and returns one line per
// failing case.
func robustSweepFailures(t *testing.T, frames int, skip func(path string, cfg *Config) bool) []string {
	var failures []string
	for _, b := range robustSweepBases {
		base := DefaultConfig()
		b.setup(&base)
		n := len(robustLeaves(&base))
		for i := 0; i < n; i++ {
			kind := robustLeaves(&base)[i].v.Kind()
			seen := map[Config]bool{base: true}
			for _, val := range robustSweepValues(kind) {
				cfg := base
				l := robustLeaves(&cfg)[i]
				l.v.Set(reflect.ValueOf(val).Convert(l.v.Type()))
				cfg.Validate()
				robustCapResources(&cfg)
				// Run each distinct validated config once (NaN fields never
				// compare equal, so those always run).
				if seen[cfg] {
					continue
				}
				seen[cfg] = true
				if skip != nil && skip(l.path, &cfg) {
					continue
				}
				if msg := robustSweepRun(cfg, b, frames); msg != "" {
					first := strings.SplitN(msg, "\n", 2)[0]
					failures = append(failures, fmt.Sprintf("%s %s=%v (after Validate %v): %s", b.name, l.path, val,
						l.v.Interface(), first))
					if testing.Verbose() {
						t.Log(msg)
					}
				}
			}
		}
	}
	return failures
}
