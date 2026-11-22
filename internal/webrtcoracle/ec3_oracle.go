//go:build webrtc_oracle || webrtc_src_oracle

package webrtcoracle

/*
#include "ec3_oracle.h"
*/
import "C"

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

// Ec3MaxBands and Ec3MaxChannels bound the recorded render block.
const (
	Ec3MaxBands    = C.EC3_MAX_BANDS
	Ec3MaxChannels = C.EC3_MAX_CHANNELS
)

func ec3FlattenValue(v reflect.Value, out []float64) []float64 {
	switch v.Kind() {
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			out = ec3FlattenValue(v.Field(i), out)
		}
	case reflect.Bool:
		if v.Bool() {
			out = append(out, 1)
		} else {
			out = append(out, 0)
		}
	case reflect.Int:
		out = append(out, float64(v.Int()))
	case reflect.Float32:
		out = append(out, v.Float())
	default:
		panic(fmt.Sprintf("ec3: unsupported config field kind %v", v.Kind()))
	}
	return out
}

// Ec3FlattenConfig returns the config leaves in declaration order, as the
// C++ side (ec3_config_fields.inc) reads them.
func Ec3FlattenConfig(cfg *aec3.EchoCanceller3Config) []float64 {
	out := ec3FlattenValue(reflect.ValueOf(cfg).Elem(), nil)
	if n := int(C.ec3_config_num_fields()); n != len(out) {
		panic(fmt.Sprintf("ec3: Go config has %d leaves, C++ list has %d", len(out), n))
	}
	return out
}

func ec3D(x []float64) *C.double {
	if x == nil {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&x[0]))
}

func ec3Fs(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

func ec3B(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// Ec3BpRecord is one BlockProcessor -> EchoRemover::ProcessCapture call as
// seen by the recording echo remover.
type Ec3BpRecord struct {
	Called                bool
	GainChange            bool
	DelayChange           int
	ClockDrift            bool
	Saturation            bool
	HasDelay              bool
	Delay                 int
	Quality               int
	BlocksSinceLastChange int
	BlocksSinceLastUpdate int
	Position              int
	RenderActivity        bool
	Block                 [Ec3MaxBands * Ec3MaxChannels * 64]float32
}

// Ec3BlockProcessor drives webrtc::BlockProcessor built with the prebuilt
// RenderDelayBuffer and RenderDelayController (or their scalar source build)
// and a recording EchoRemover that halves the capture block and copies its
// lowest band to the linear output.
type Ec3BlockProcessor struct {
	p      unsafe.Pointer
	scalar bool
}

// NewEc3BlockProcessor creates the C++ block processor; scalar selects the
// Aec3Optimization::kNone / scalar-FFT source build.
func NewEc3BlockProcessor(cfg *aec3.EchoCanceller3Config, rate, numRender, numCapture int,
	scalar bool) *Ec3BlockProcessor {
	flat := Ec3FlattenConfig(cfg)
	b := &Ec3BlockProcessor{scalar: scalar}
	if scalar {
		b.p = C.ec3s_bp_new(ec3D(flat), C.int(rate), C.int(numRender), C.int(numCapture))
	} else {
		b.p = C.ec3_bp_new(ec3D(flat), C.int(rate), C.int(numRender), C.int(numCapture))
	}
	return b
}

func (b *Ec3BlockProcessor) Free() {
	if b.scalar {
		C.ec3s_bp_free(b.p)
	} else {
		C.ec3_bp_free(b.p)
	}
}

// BufferRender buffers a [band][channel][64] block.
func (b *Ec3BlockProcessor) BufferRender(block []float32) {
	if b.scalar {
		C.ec3s_bp_buffer_render(b.p, ec3Fs(block))
	} else {
		C.ec3_bp_buffer_render(b.p, ec3Fs(block))
	}
}

// ProcessCapture processes a [band][channel][64] capture block in place;
// linear ([channel][64]) receives the linear output when exported.
func (b *Ec3BlockProcessor) ProcessCapture(gainChange, saturation bool, capture, linear []float32) Ec3BpRecord {
	var rec C.ec3_bp_record
	if b.scalar {
		C.ec3s_bp_process_capture(b.p, ec3B(gainChange), ec3B(saturation), ec3Fs(capture), ec3Fs(linear), &rec)
	} else {
		C.ec3_bp_process_capture(b.p, ec3B(gainChange), ec3B(saturation), ec3Fs(capture), ec3Fs(linear), &rec)
	}
	out := Ec3BpRecord{
		Called:                rec.called != 0,
		GainChange:            rec.gain_change != 0,
		DelayChange:           int(rec.delay_change),
		ClockDrift:            rec.clock_drift != 0,
		Saturation:            rec.saturation != 0,
		HasDelay:              rec.has_delay != 0,
		Delay:                 int(rec.delay),
		Quality:               int(rec.quality),
		BlocksSinceLastChange: int(rec.blocks_since_last_change),
		BlocksSinceLastUpdate: int(rec.blocks_since_last_update),
		Position:              int(rec.position),
		RenderActivity:        rec.render_activity != 0,
	}
	for i := range out.Block {
		out.Block[i] = float32(rec.block[i])
	}
	return out
}

func (b *Ec3BlockProcessor) SetAudioBufferDelay(delayMs int) {
	if b.scalar {
		C.ec3s_bp_set_audio_buffer_delay(b.p, C.int(delayMs))
	} else {
		C.ec3_bp_set_audio_buffer_delay(b.p, C.int(delayMs))
	}
}

// Metrics returns BlockProcessor::GetMetrics.
func (b *Ec3BlockProcessor) Metrics() aec3.EchoControlMetrics {
	var erl, erle C.double
	var delay C.int
	if b.scalar {
		C.ec3s_bp_metrics(b.p, &erl, &erle, &delay)
	} else {
		C.ec3_bp_metrics(b.p, &erl, &erle, &delay)
	}
	return aec3.EchoControlMetrics{EchoReturnLoss: float64(erl), EchoReturnLossEnhancement: float64(erle),
		DelayMs: int(delay)}
}

// Ec3Canceller drives webrtc::EchoCanceller3 through webrtc::AudioBuffer the
// way AudioProcessingImpl does.
type Ec3Canceller struct {
	p      unsafe.Pointer
	scalar bool
}

// NewEc3Canceller creates the C++ echo canceller; mc is the optional
// multichannel config.
func NewEc3Canceller(cfg, mc *aec3.EchoCanceller3Config, rate, numRender, numCapture int, scalar bool) *Ec3Canceller {
	flat := Ec3FlattenConfig(cfg)
	var flatMc []float64
	if mc != nil {
		flatMc = Ec3FlattenConfig(mc)
	}
	c := &Ec3Canceller{scalar: scalar}
	if scalar {
		c.p = C.ec3s_canceller_new(ec3D(flat), ec3D(flatMc), C.int(rate), C.int(numRender), C.int(numCapture))
	} else {
		c.p = C.ec3_canceller_new(ec3D(flat), ec3D(flatMc), C.int(rate), C.int(numRender), C.int(numCapture))
	}
	return c
}

func (c *Ec3Canceller) Free() {
	if c.scalar {
		C.ec3s_canceller_free(c.p)
	} else {
		C.ec3_canceller_free(c.p)
	}
}

// AnalyzeRender takes a FloatS16 full-band frame, [channel][rate/100].
func (c *Ec3Canceller) AnalyzeRender(render []float32) {
	if c.scalar {
		C.ec3s_canceller_analyze_render(c.p, ec3Fs(render))
	} else {
		C.ec3_canceller_analyze_render(c.p, ec3Fs(render))
	}
}

// ProcessCapture processes a FloatS16 full-band frame in place; linear
// ([channel][160], may be nil) receives the FloatS16 linear output.
func (c *Ec3Canceller) ProcessCapture(capture, linear []float32, levelChange bool) {
	if c.scalar {
		C.ec3s_canceller_process_capture(c.p, ec3Fs(capture), ec3Fs(linear), ec3B(levelChange))
	} else {
		C.ec3_canceller_process_capture(c.p, ec3Fs(capture), ec3Fs(linear), ec3B(levelChange))
	}
}

// AnalyzeRenderFloat takes a [-1, 1] frame through AudioBuffer::CopyFrom.
func (c *Ec3Canceller) AnalyzeRenderFloat(render []float32) {
	if c.scalar {
		C.ec3s_canceller_analyze_render_float(c.p, ec3Fs(render))
	} else {
		C.ec3_canceller_analyze_render_float(c.p, ec3Fs(render))
	}
}

// ProcessCaptureFloat processes a [-1, 1] frame through
// AudioBuffer::CopyFrom/CopyTo; linear receives GetLinearAecOutput.
func (c *Ec3Canceller) ProcessCaptureFloat(capture, linear []float32, levelChange bool) {
	if c.scalar {
		C.ec3s_canceller_process_capture_float(c.p, ec3Fs(capture), ec3Fs(linear), ec3B(levelChange))
	} else {
		C.ec3_canceller_process_capture_float(c.p, ec3Fs(capture), ec3Fs(linear), ec3B(levelChange))
	}
}

// Metrics returns EchoCanceller3::GetMetrics.
func (c *Ec3Canceller) Metrics() aec3.EchoControlMetrics {
	var erl, erle C.double
	var delay C.int
	if c.scalar {
		C.ec3s_canceller_metrics(c.p, &erl, &erle, &delay)
	} else {
		C.ec3_canceller_metrics(c.p, &erl, &erle, &delay)
	}
	return aec3.EchoControlMetrics{EchoReturnLoss: float64(erl), EchoReturnLossEnhancement: float64(erle),
		DelayMs: int(delay)}
}

func (c *Ec3Canceller) SetAudioBufferDelay(delayMs int) {
	if c.scalar {
		C.ec3s_canceller_set_audio_buffer_delay(c.p, C.int(delayMs))
	} else {
		C.ec3_canceller_set_audio_buffer_delay(c.p, C.int(delayMs))
	}
}

func (c *Ec3Canceller) SetCaptureOutputUsage(used bool) {
	if c.scalar {
		C.ec3s_canceller_set_capture_output_usage(c.p, ec3B(used))
	} else {
		C.ec3_canceller_set_capture_output_usage(c.p, ec3B(used))
	}
}
