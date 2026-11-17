//go:build webrtc_oracle

package webrtcoracle

/*
#include "dly_oracle.h"
*/
import "C"

import (
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

func dlyF(p []float32) *C.float {
	if len(p) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&p[0]))
}

func dlyB(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func dlyConfig(c *aec3.EchoCanceller3Config) C.dly_config {
	d := &c.Delay
	return C.dly_config{
		default_delay:                           C.int(d.DefaultDelay),
		down_sampling_factor:                    C.int(d.DownSamplingFactor),
		num_filters:                             C.int(d.NumFilters),
		delay_headroom_samples:                  C.int(d.DelayHeadroomSamples),
		hysteresis_limit_blocks:                 C.int(d.HysteresisLimitBlocks),
		delay_estimate_smoothing:                C.float(d.DelayEstimateSmoothing),
		delay_estimate_smoothing_delay_found:    C.float(d.DelayEstimateSmoothingDelayFound),
		delay_candidate_detection_threshold:     C.float(d.DelayCandidateDetectionThreshold),
		thresholds_initial:                      C.int(d.DelaySelectionThresholds.Initial),
		thresholds_converged:                    C.int(d.DelaySelectionThresholds.Converged),
		use_external_delay_estimator:            dlyB(d.UseExternalDelayEstimator),
		render_mix_downmix:                      dlyB(d.RenderAlignmentMixing.Downmix),
		render_mix_adaptive:                     dlyB(d.RenderAlignmentMixing.AdaptiveSelection),
		render_mix_threshold:                    C.float(d.RenderAlignmentMixing.ActivityPowerThreshold),
		render_mix_prefer_first_two:             dlyB(d.RenderAlignmentMixing.PreferFirstTwoChannels),
		capture_mix_downmix:                     dlyB(d.CaptureAlignmentMixing.Downmix),
		capture_mix_adaptive:                    dlyB(d.CaptureAlignmentMixing.AdaptiveSelection),
		capture_mix_threshold:                   C.float(d.CaptureAlignmentMixing.ActivityPowerThreshold),
		capture_mix_prefer_first_two:            dlyB(d.CaptureAlignmentMixing.PreferFirstTwoChannels),
		detect_pre_echo:                         dlyB(d.DetectPreEcho),
		refined_length_blocks:                   C.int(c.Filter.Refined.LengthBlocks),
		active_render_limit:                     C.float(c.RenderLevels.ActiveRenderLimit),
		poor_excitation_render_limit:            C.float(c.RenderLevels.PoorExcitationRenderLimit),
		poor_excitation_render_limit_ds8:        C.float(c.RenderLevels.PoorExcitationRenderLimitDs8),
		render_power_gain_db:                    C.float(c.RenderLevels.RenderPowerGainDb),
		excess_render_detection_interval_blocks: C.int(c.Buffering.ExcessRenderDetectionIntervalBlocks),
		max_allowed_excess_render_blocks:        C.int(c.Buffering.MaxAllowedExcessRenderBlocks),
	}
}

// DlyMatchedFilterCoreLane selects the C++ MatchedFilterCore variant.
type DlyMatchedFilterCoreLane int

const (
	DlyCoreScalar DlyMatchedFilterCoreLane = iota
	DlyCoreNeon
	DlyCoreNeonAccumulatedError
)

// DlyMatchedFilterCore runs one C++ MatchedFilterCore variant in place on h,
// errorSum, filtersUpdated and accumulatedError.
func DlyMatchedFilterCore(lane DlyMatchedFilterCoreLane, xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool,
	accumulatedError, scratch []float32) {
	upd := dlyB(*filtersUpdated)
	es := C.float(*errorSum)
	C.dly_mf_core(C.int(lane), C.size_t(xStartIndex), C.float(x2SumThreshold), C.float(smoothing),
		dlyF(x), C.int(len(x)), dlyF(y), C.int(len(y)), dlyF(h), C.int(len(h)), &upd, &es,
		dlyB(computeAccumulatedError), dlyF(accumulatedError), C.int(len(accumulatedError)),
		dlyF(scratch), C.int(len(scratch)))
	*filtersUpdated = upd != 0
	*errorSum = float32(es)
}

// DlyMatchedFilter wraps webrtc::MatchedFilter with its own
// DownsampledRenderBuffer.
type DlyMatchedFilter struct{ p unsafe.Pointer }

func NewDlyMatchedFilter(opt aec3.Aec3Optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters,
	alignmentShiftSubBlocks int, excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold float32,
	detectPreEcho bool, renderBufferSize int) *DlyMatchedFilter {
	cOpt := map[aec3.Aec3Optimization]int{aec3.OptNone: 0, aec3.OptSse2: 1, aec3.OptAvx2: 2, aec3.OptNeon: 3}[opt]
	return &DlyMatchedFilter{C.dly_mf_new(C.int(cOpt), C.int(subBlockSize), C.int(windowSizeSubBlocks),
		C.int(numMatchedFilters), C.int(alignmentShiftSubBlocks), C.float(excitationLimit), C.float(smoothingFast),
		C.float(smoothingSlow), C.float(matchingFilterThreshold), dlyB(detectPreEcho), C.int(renderBufferSize))}
}

func (m *DlyMatchedFilter) Free()                { C.dly_mf_free(m.p) }
func (m *DlyMatchedFilter) Reset(full bool)      { C.dly_mf_reset(m.p, dlyB(full)) }
func (m *DlyMatchedFilter) GetMaxFilterLag() int { return int(C.dly_mf_max_filter_lag(m.p)) }

func (m *DlyMatchedFilter) Update(rb *aec3.DownsampledRenderBuffer, capture []float32, slow bool) (aec3.MatchedFilterLagEstimate, bool) {
	var lag, pre C.int
	ok := C.dly_mf_update(m.p, dlyF(rb.Buffer), C.int(rb.Read), C.int(rb.Write), dlyF(capture), C.int(len(capture)),
		dlyB(slow), &lag, &pre)
	return aec3.MatchedFilterLagEstimate{Lag: int(lag), PreEchoLag: int(pre)}, ok != 0
}

// DlyDecimator wraps webrtc::Decimator.
type DlyDecimator struct{ p unsafe.Pointer }

func NewDlyDecimator(factor int) *DlyDecimator {
	return &DlyDecimator{C.dly_decimator_new(C.int(factor))}
}
func (d *DlyDecimator) Free() { C.dly_decimator_free(d.p) }
func (d *DlyDecimator) Decimate(in, out []float32) {
	C.dly_decimator_decimate(d.p, dlyF(in), dlyF(out), C.int(len(out)))
}

// DlyFrameBlocker wraps webrtc::FrameBlocker; frames and blocks are flat
// [band][channel][sample].
type DlyFrameBlocker struct{ p unsafe.Pointer }

func NewDlyFrameBlocker(bands, channels int) *DlyFrameBlocker {
	return &DlyFrameBlocker{C.dly_frame_blocker_new(C.int(bands), C.int(channels))}
}
func (f *DlyFrameBlocker) Free() { C.dly_frame_blocker_free(f.p) }
func (f *DlyFrameBlocker) InsertSubFrameAndExtractBlock(subFrame, block []float32) {
	C.dly_frame_blocker_insert(f.p, dlyF(subFrame), dlyF(block))
}
func (f *DlyFrameBlocker) IsBlockAvailable() bool { return C.dly_frame_blocker_available(f.p) != 0 }
func (f *DlyFrameBlocker) ExtractBlock(block []float32) {
	C.dly_frame_blocker_extract(f.p, dlyF(block))
}

// DlyBlockFramer wraps webrtc::BlockFramer.
type DlyBlockFramer struct{ p unsafe.Pointer }

func NewDlyBlockFramer(bands, channels int) *DlyBlockFramer {
	return &DlyBlockFramer{C.dly_block_framer_new(C.int(bands), C.int(channels))}
}
func (f *DlyBlockFramer) Free()                       { C.dly_block_framer_free(f.p) }
func (f *DlyBlockFramer) InsertBlock(block []float32) { C.dly_block_framer_insert(f.p, dlyF(block)) }
func (f *DlyBlockFramer) InsertBlockAndExtractSubFrame(block, subFrame []float32) {
	C.dly_block_framer_insert_extract(f.p, dlyF(block), dlyF(subFrame))
}

// DlyBlockDelayBuffer wraps webrtc::BlockDelayBuffer over an AudioBuffer.
type DlyBlockDelayBuffer struct{ p unsafe.Pointer }

func NewDlyBlockDelayBuffer(channels, sampleRateHz, delaySamples int) *DlyBlockDelayBuffer {
	return &DlyBlockDelayBuffer{C.dly_block_delay_buffer_new(C.int(channels), C.int(sampleRateHz), C.int(delaySamples))}
}
func (b *DlyBlockDelayBuffer) Free() { C.dly_block_delay_buffer_free(b.p) }

// DelaySignal delays a flat [band][channel][160] frame in place.
func (b *DlyBlockDelayBuffer) DelaySignal(frame []float32) {
	C.dly_block_delay_buffer_delay(b.p, dlyF(frame))
}

// The class wrappers below run the prebuilt library classes (the NEON lane
// DetectOptimization selects on arm64) in the default build and the scalar
// source build under purego.

// DlyEchoPathDelayEstimator wraps webrtc::EchoPathDelayEstimator.
type DlyEchoPathDelayEstimator struct{ p unsafe.Pointer }

func NewDlyEchoPathDelayEstimator(c *aec3.EchoCanceller3Config, captureChannels, renderBufferSize int) *DlyEchoPathDelayEstimator {
	cfg := dlyConfig(c)
	if purego {
		return &DlyEchoPathDelayEstimator{C.dly_scalar_epde_new(&cfg, C.int(captureChannels), C.int(renderBufferSize))}
	}
	return &DlyEchoPathDelayEstimator{C.dly_epde_new(&cfg, C.int(captureChannels), C.int(renderBufferSize))}
}

func (e *DlyEchoPathDelayEstimator) Free() {
	if purego {
		C.dly_scalar_epde_free(e.p)
	} else {
		C.dly_epde_free(e.p)
	}
}

func (e *DlyEchoPathDelayEstimator) Reset(resetDelayConfidence bool) {
	if purego {
		C.dly_scalar_epde_reset(e.p, dlyB(resetDelayConfidence))
	} else {
		C.dly_epde_reset(e.p, dlyB(resetDelayConfidence))
	}
}

// EstimateDelay copies rb into the C++ buffer and runs EstimateDelay.
func (e *DlyEchoPathDelayEstimator) EstimateDelay(rb *aec3.DownsampledRenderBuffer, capture []float32, bands,
	channels int) (est aec3.DelayEstimate, ok bool, clockdrift int) {
	var q, d, cd C.int
	var r C.int
	if purego {
		r = C.dly_scalar_epde_estimate(e.p, dlyF(rb.Buffer), C.int(rb.Read), C.int(rb.Write), dlyF(capture),
			C.int(bands), C.int(channels), &q, &d, &cd)
	} else {
		r = C.dly_epde_estimate(e.p, dlyF(rb.Buffer), C.int(rb.Read), C.int(rb.Write), dlyF(capture),
			C.int(bands), C.int(channels), &q, &d, &cd)
	}
	return aec3.DelayEstimate{Quality: aec3.DelayEstimateQuality(q), Delay: int(d)}, r != 0, int(cd)
}

// DlyRenderDelayBuffer wraps webrtc::RenderDelayBuffer.
type DlyRenderDelayBuffer struct{ p unsafe.Pointer }

func NewDlyRenderDelayBuffer(c *aec3.EchoCanceller3Config, sampleRateHz, channels int) *DlyRenderDelayBuffer {
	cfg := dlyConfig(c)
	if purego {
		return &DlyRenderDelayBuffer{C.dly_scalar_rdb_new(&cfg, C.int(sampleRateHz), C.int(channels))}
	}
	return &DlyRenderDelayBuffer{C.dly_rdb_new(&cfg, C.int(sampleRateHz), C.int(channels))}
}

func (r *DlyRenderDelayBuffer) Free() {
	if purego {
		C.dly_scalar_rdb_free(r.p)
	} else {
		C.dly_rdb_free(r.p)
	}
}

func (r *DlyRenderDelayBuffer) Reset() {
	if purego {
		C.dly_scalar_rdb_reset(r.p)
	} else {
		C.dly_rdb_reset(r.p)
	}
}

func (r *DlyRenderDelayBuffer) Insert(block []float32, bands, channels int) aec3.BufferingEvent {
	if purego {
		return aec3.BufferingEvent(C.dly_scalar_rdb_insert(r.p, dlyF(block), C.int(bands), C.int(channels)))
	}
	return aec3.BufferingEvent(C.dly_rdb_insert(r.p, dlyF(block), C.int(bands), C.int(channels)))
}

func (r *DlyRenderDelayBuffer) PrepareCaptureProcessing() aec3.BufferingEvent {
	if purego {
		return aec3.BufferingEvent(C.dly_scalar_rdb_prepare(r.p))
	}
	return aec3.BufferingEvent(C.dly_rdb_prepare(r.p))
}

func (r *DlyRenderDelayBuffer) HandleSkippedCaptureProcessing() {
	if purego {
		C.dly_scalar_rdb_skipped(r.p)
	} else {
		C.dly_rdb_skipped(r.p)
	}
}

func (r *DlyRenderDelayBuffer) AlignFromDelay(delay int) bool {
	if purego {
		return C.dly_scalar_rdb_align_from_delay(r.p, C.int(delay)) != 0
	}
	return C.dly_rdb_align_from_delay(r.p, C.int(delay)) != 0
}

func (r *DlyRenderDelayBuffer) AlignFromExternalDelay() {
	if purego {
		C.dly_scalar_rdb_align_from_external_delay(r.p)
	} else {
		C.dly_rdb_align_from_external_delay(r.p)
	}
}

// Delay returns the C++ size_t Delay() reinterpreted as a signed value.
func (r *DlyRenderDelayBuffer) Delay() int {
	if purego {
		return int(C.dly_scalar_rdb_delay(r.p))
	}
	return int(C.dly_rdb_delay(r.p))
}

func (r *DlyRenderDelayBuffer) MaxDelay() int {
	if purego {
		return int(C.dly_scalar_rdb_max_delay(r.p))
	}
	return int(C.dly_rdb_max_delay(r.p))
}

func (r *DlyRenderDelayBuffer) SetAudioBufferDelay(ms int) {
	if purego {
		C.dly_scalar_rdb_set_audio_buffer_delay(r.p, C.int(ms))
	} else {
		C.dly_rdb_set_audio_buffer_delay(r.p, C.int(ms))
	}
}

func (r *DlyRenderDelayBuffer) HasReceivedBufferDelay() bool {
	if purego {
		return C.dly_scalar_rdb_has_received_buffer_delay(r.p) != 0
	}
	return C.dly_rdb_has_received_buffer_delay(r.p) != 0
}

// DlyRenderDelayBufferState is a snapshot of the C++ buffer contents.
type DlyRenderDelayBufferState struct {
	Size, Bands, Channels, LowRateSize int
	Blocks                             []float32 // [slot][band][channel][64]
	Spectra, FftRe, FftIm              []float32 // [slot][channel][65]
	LowRate                            []float32
	// BlocksRead, BlocksWrite, SpectraRead, SpectraWrite, FftPosition,
	// Headroom, LowRateRead, LowRateWrite, RenderActivity.
	Idx [9]int
}

func (r *DlyRenderDelayBuffer) State() *DlyRenderDelayBufferState {
	var sizes [4]C.int
	if purego {
		C.dly_scalar_rdb_sizes(r.p, &sizes[0])
	} else {
		C.dly_rdb_sizes(r.p, &sizes[0])
	}
	s := &DlyRenderDelayBufferState{Size: int(sizes[0]), Bands: int(sizes[1]), Channels: int(sizes[2]),
		LowRateSize: int(sizes[3])}
	s.Blocks = make([]float32, s.Size*s.Bands*s.Channels*aec3.BlockSize)
	s.Spectra = make([]float32, s.Size*s.Channels*aec3.FftLengthBy2Plus1)
	s.FftRe = make([]float32, len(s.Spectra))
	s.FftIm = make([]float32, len(s.Spectra))
	s.LowRate = make([]float32, s.LowRateSize)
	var idx [9]C.int
	if purego {
		C.dly_scalar_rdb_state(r.p, dlyF(s.Blocks), dlyF(s.Spectra), dlyF(s.FftRe), dlyF(s.FftIm), dlyF(s.LowRate), &idx[0])
	} else {
		C.dly_rdb_state(r.p, dlyF(s.Blocks), dlyF(s.Spectra), dlyF(s.FftRe), dlyF(s.FftIm), dlyF(s.LowRate), &idx[0])
	}
	for i := range idx {
		s.Idx[i] = int(idx[i])
	}
	return s
}

// DlyRenderDelayController wraps webrtc::RenderDelayController.
type DlyRenderDelayController struct{ p unsafe.Pointer }

func NewDlyRenderDelayController(c *aec3.EchoCanceller3Config, sampleRateHz, channels int) *DlyRenderDelayController {
	cfg := dlyConfig(c)
	if purego {
		return &DlyRenderDelayController{C.dly_scalar_rdc_new(&cfg, C.int(sampleRateHz), C.int(channels))}
	}
	return &DlyRenderDelayController{C.dly_rdc_new(&cfg, C.int(sampleRateHz), C.int(channels))}
}

func (c *DlyRenderDelayController) Free() {
	if purego {
		C.dly_scalar_rdc_free(c.p)
	} else {
		C.dly_rdc_free(c.p)
	}
}

func (c *DlyRenderDelayController) Reset(resetDelayConfidence bool) {
	if purego {
		C.dly_scalar_rdc_reset(c.p, dlyB(resetDelayConfidence))
	} else {
		C.dly_rdc_reset(c.p, dlyB(resetDelayConfidence))
	}
}

// GetDelay runs GetDelay with rdb's downsampled buffer and Delay().
func (c *DlyRenderDelayController) GetDelay(rdb *DlyRenderDelayBuffer, capture []float32, bands, channels int) (aec3.DelayEstimate, bool) {
	var out [4]C.int
	var r C.int
	if purego {
		r = C.dly_scalar_rdc_get_delay(c.p, rdb.p, dlyF(capture), C.int(bands), C.int(channels), &out[0])
	} else {
		r = C.dly_rdc_get_delay(c.p, rdb.p, dlyF(capture), C.int(bands), C.int(channels), &out[0])
	}
	return aec3.DelayEstimate{Quality: aec3.DelayEstimateQuality(out[0]), Delay: int(out[1]),
		BlocksSinceLastChange: int(out[2]), BlocksSinceLastUpdate: int(out[3])}, r != 0
}

func (c *DlyRenderDelayController) HasClockdrift() bool {
	if purego {
		return C.dly_scalar_rdc_has_clockdrift(c.p) != 0
	}
	return C.dly_rdc_has_clockdrift(c.p) != 0
}

// DlyAlignmentMixer wraps webrtc::AlignmentMixer.
type DlyAlignmentMixer struct {
	p        unsafe.Pointer
	channels int
}

func NewDlyAlignmentMixer(channels int, downmix, adaptive bool, threshold float32, preferFirstTwo bool) *DlyAlignmentMixer {
	return &DlyAlignmentMixer{C.dly_alignment_mixer_new(C.int(channels), dlyB(downmix), dlyB(adaptive),
		C.float(threshold), dlyB(preferFirstTwo)), channels}
}
func (m *DlyAlignmentMixer) Free() { C.dly_alignment_mixer_free(m.p) }

// ProduceOutput mixes a flat [channel][64] block into y.
func (m *DlyAlignmentMixer) ProduceOutput(block, y []float32) {
	C.dly_alignment_mixer_produce(m.p, dlyF(block), C.int(m.channels), dlyF(y))
}

// DlyClockdriftDetector wraps webrtc::ClockdriftDetector.
type DlyClockdriftDetector struct{ p unsafe.Pointer }

func NewDlyClockdriftDetector() *DlyClockdriftDetector {
	return &DlyClockdriftDetector{C.dly_clockdrift_new()}
}
func (d *DlyClockdriftDetector) Free() { C.dly_clockdrift_free(d.p) }

// Update feeds one estimate and returns the resulting level.
func (d *DlyClockdriftDetector) Update(delay int) int {
	return int(C.dly_clockdrift_update(d.p, C.int(delay)))
}
