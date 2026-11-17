//go:build webrtc_oracle

package webrtcoracle

/*
#include "sgain_oracle.h"
*/
import "C"

import (
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

func sgainF(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

func sgainSpec(x [][aec3.FftLengthBy2Plus1]float32) *C.float {
	return (*C.float)(unsafe.Pointer(&x[0][0]))
}

func sgainB(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func sgainTuning(t *aec3.Tuning) [8]C.float {
	return [8]C.float{
		C.float(t.MaskLf.EnrTransparent), C.float(t.MaskLf.EnrSuppress), C.float(t.MaskLf.EmrTransparent),
		C.float(t.MaskHf.EnrTransparent), C.float(t.MaskHf.EnrSuppress), C.float(t.MaskHf.EmrTransparent),
		C.float(t.MaxIncFactor), C.float(t.MaxDecFactorLf),
	}
}

func sgainCConfig(c *aec3.EchoCanceller3Config) *C.sg_config {
	s := &c.Suppressor
	dn := &s.DominantNearendDetection
	sn := &s.SubbandNearendDetection
	hb := &s.HighBandsSuppression
	return &C.sg_config{
		st:                                   *stCConfig(c),
		config_change_duration_blocks:        C.int(c.Filter.ConfigChangeDurationBlocks),
		low_render_limit:                     C.float(c.EchoAudibility.LowRenderLimit),
		normal_render_limit:                  C.float(c.EchoAudibility.NormalRenderLimit),
		floor_power:                          C.float(c.EchoAudibility.FloorPower),
		audibility_threshold_lf:              C.float(c.EchoAudibility.AudibilityThresholdLf),
		audibility_threshold_mf:              C.float(c.EchoAudibility.AudibilityThresholdMf),
		audibility_threshold_hf:              C.float(c.EchoAudibility.AudibilityThresholdHf),
		nearend_average_blocks:               C.int(s.NearendAverageBlocks),
		normal_tuning:                        sgainTuning(&s.NormalTuning),
		nearend_tuning:                       sgainTuning(&s.NearendTuning),
		lf_smoothing_during_initial_phase:    sgainB(s.LfSmoothingDuringInitialPhase),
		last_permanent_lf_smoothing_band:     C.int(s.LastPermanentLfSmoothingBand),
		last_lf_smoothing_band:               C.int(s.LastLfSmoothingBand),
		last_lf_band:                         C.int(s.LastLfBand),
		first_hf_band:                        C.int(s.FirstHfBand),
		dn_enr_threshold:                     C.float(dn.EnrThreshold),
		dn_enr_exit_threshold:                C.float(dn.EnrExitThreshold),
		dn_snr_threshold:                     C.float(dn.SnrThreshold),
		dn_hold_duration:                     C.int(dn.HoldDuration),
		dn_trigger_threshold:                 C.int(dn.TriggerThreshold),
		dn_use_during_initial_phase:          sgainB(dn.UseDuringInitialPhase),
		dn_use_unbounded_echo_spectrum:       sgainB(dn.UseUnboundedEchoSpectrum),
		sn_nearend_average_blocks:            C.int(sn.NearendAverageBlocks),
		sn_subband1_low:                      C.int(sn.Subband1.Low),
		sn_subband1_high:                     C.int(sn.Subband1.High),
		sn_subband2_low:                      C.int(sn.Subband2.Low),
		sn_subband2_high:                     C.int(sn.Subband2.High),
		sn_nearend_threshold:                 C.float(sn.NearendThreshold),
		sn_snr_threshold:                     C.float(sn.SnrThreshold),
		use_subband_nearend_detection:        sgainB(s.UseSubbandNearendDetection),
		hb_enr_threshold:                     C.float(hb.EnrThreshold),
		hb_max_gain_during_echo:              C.float(hb.MaxGainDuringEcho),
		hb_anti_howling_activation_threshold: C.float(hb.AntiHowlingActivationThreshold),
		hb_anti_howling_gain:                 C.float(hb.AntiHowlingGain),
		floor_first_increase:                 C.float(s.FloorFirstIncrease),
		conservative_hf_suppression:          sgainB(s.ConservativeHfSuppression),
	}
}

// SgainHarness pairs a C++ {render buffer, AecState, RenderSignalAnalyzer,
// SuppressionGain} with the identical Go objects. Every setter drives both.
type SgainHarness struct {
	c         unsafe.Pointer
	NumRender int
	NumCap    int
	Size      int
	Blocks    *aec3.BlockBuffer
	Spectra   *aec3.SpectrumBuffer
	Ffts      *aec3.FftBuffer
	Buffer    *aec3.RenderBuffer
	Aec       *aec3.AecState
	Rsa       *aec3.RenderSignalAnalyzer
	Gain      *aec3.SuppressionGain
}

func NewSgainHarness(cfg *aec3.EchoCanceller3Config, opt aec3.Aec3Optimization, rate, numRender, numCapture,
	size int) *SgainHarness {
	h := &SgainHarness{
		c: C.sgain_new(sgainCConfig(cfg), C.int(opt), C.int(rate), C.int(numRender), C.int(numCapture),
			C.int(size)),
		NumRender: numRender,
		NumCap:    numCapture,
		Size:      size,
		Blocks:    aec3.NewBlockBuffer(size, 1, numRender),
		Spectra:   aec3.NewSpectrumBuffer(size, numRender),
		Ffts:      aec3.NewFftBuffer(size, numRender),
		Aec:       aec3.NewAecState(cfg, numCapture),
		Rsa:       aec3.NewRenderSignalAnalyzer(cfg),
		Gain:      aec3.NewSuppressionGain(cfg, opt, rate, numCapture),
	}
	h.Buffer = aec3.NewRenderBuffer(h.Blocks, h.Spectra, h.Ffts)
	return h
}

func (h *SgainHarness) Free() { C.sgain_free(h.c) }

func (h *SgainHarness) SetRenderBlock(slot, ch int, x []float32) {
	copy(h.Blocks.Buffer[slot].View(0, ch), x)
	C.sgain_set_render_block(h.c, C.int(slot), C.int(ch), sgainF(x))
}

func (h *SgainHarness) SetRenderSpectrum(slot, ch int, x2 *[aec3.FftLengthBy2Plus1]float32) {
	h.Spectra.Buffer[slot][ch] = *x2
	C.sgain_set_render_spectrum(h.c, C.int(slot), C.int(ch), (*C.float)(unsafe.Pointer(&x2[0])))
}

func (h *SgainHarness) SetIndices(write, read int) {
	h.Blocks.Write, h.Spectra.Write, h.Ffts.Write = write, write, write
	h.Blocks.Read, h.Spectra.Read, h.Ffts.Read = read, read, read
	C.sgain_set_indices(h.c, C.int(write), C.int(read))
}

func (h *SgainHarness) HandleEchoPathChange(v aec3.EchoPathVariability) {
	h.Aec.HandleEchoPathChange(v)
	C.sgain_aec_handle_echo_path_change(h.c, sgainB(v.GainChange), C.int(v.DelayChange))
}

func (h *SgainHarness) UpdateCaptureSaturation(s bool) {
	h.Aec.UpdateCaptureSaturation(s)
	C.sgain_aec_update_capture_saturation(h.c, sgainB(s))
}

// AecUpdate runs AecState::Update on both sides.
func (h *SgainHarness) AecUpdate(ext *aec3.DelayEstimate, fr [][][aec3.FftLengthBy2Plus1]float32, ir [][]float32,
	E2, Y2 [][aec3.FftLengthBy2Plus1]float32, so []aec3.SubtractorOutput) {
	h.Aec.Update(ext, fr, ir, h.Buffer, E2, Y2, so)
	var extValid, extQuality, extDelay C.int
	if ext != nil {
		extValid, extQuality, extDelay = 1, C.int(ext.Quality), C.int(ext.Delay)
	}
	flatFr := stFlatFreqResp(fr)
	flatIr := stFlat(ir)
	sub := make([]float32, 0, 5*len(so))
	for _, s := range so {
		sub = append(sub, s.Y2, s.E2RefinedSum, s.E2CoarseSum, s.SRefinedMaxAbs, s.SCoarseMaxAbs)
	}
	C.sgain_aec_update(h.c, extValid, extQuality, extDelay, sgainF(flatFr), C.int(len(fr[0])), sgainF(flatIr),
		C.int(len(ir[0])), sgainSpec(E2), sgainSpec(Y2), sgainF(sub))
}

// CSaturatedEcho and CMinDirectPathFilterDelay read the C++ AecState.
func (h *SgainHarness) CSaturatedEcho() bool { return C.sgain_aec_saturated_echo(h.c) != 0 }
func (h *SgainHarness) CMinDirectPathFilterDelay() int {
	return int(C.sgain_aec_min_direct_path_filter_delay(h.c))
}

// RsaUpdate runs RenderSignalAnalyzer::Update on both sides.
func (h *SgainHarness) RsaUpdate(delay int, hasDelay bool) {
	h.Rsa.Update(h.Buffer, delay, hasDelay)
	C.sgain_rsa_update(h.c, C.int(delay), sgainB(hasDelay))
}

// CNarrowPeakBand reads the C++ RenderSignalAnalyzer.
func (h *SgainHarness) CNarrowPeakBand() (int, bool) {
	var band C.int
	ok := C.sgain_rsa_narrow_peak_band(h.c, &band) != 0
	return int(band), ok
}

func (h *SgainHarness) SetInitialState(s bool) {
	h.Gain.SetInitialState(s)
	C.sgain_set_initial_state(h.c, sgainB(s))
}

// CGetGain runs the C++ SuppressionGain::GetGain. render is
// [band][channel][64] flattened with NumRender channels.
func (h *SgainHarness) CGetGain(nearend, echo, residual, residualUnbounded, comfortNoise [][aec3.FftLengthBy2Plus1]float32,
	render []float32, numBands int, clockDrift bool, low *[aec3.FftLengthBy2Plus1]float32) (high float32,
	dominantNearend bool) {
	var hg C.float
	dn := C.sgain_get_gain(h.c, sgainSpec(nearend), sgainSpec(echo), sgainSpec(residual), sgainSpec(residualUnbounded),
		sgainSpec(comfortNoise), sgainF(render), C.int(numBands), sgainB(clockDrift), &hg,
		(*C.float)(unsafe.Pointer(&low[0])))
	return float32(hg), dn != 0
}
