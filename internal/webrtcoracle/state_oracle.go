//go:build webrtc_oracle

package webrtcoracle

/*
#include "state_oracle.h"
*/
import "C"

import (
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

func stF(p *float32) *C.float { return (*C.float)(unsafe.Pointer(p)) }

func stFs(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0]))
}

func stSpec(x [][aec3.FftLengthBy2Plus1]float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return (*C.float)(unsafe.Pointer(&x[0][0]))
}

func stB(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func stBools(x []bool) []C.int {
	out := make([]C.int, len(x))
	for i, b := range x {
		out[i] = stB(b)
	}
	return out
}

func stInts(x []C.int) *C.int { return (*C.int)(unsafe.Pointer(&x[0])) }

// stFlatFreqResp flattens [ch][block][65] into one buffer.
func stFlatFreqResp(fr [][][aec3.FftLengthBy2Plus1]float32) []float32 {
	var out []float32
	for _, ch := range fr {
		for _, b := range ch {
			out = append(out, b[:]...)
		}
	}
	return out
}

func stFlat(x [][]float32) []float32 {
	var out []float32
	for _, v := range x {
		out = append(out, v...)
	}
	return out
}

func stCConfig(c *aec3.EchoCanceller3Config) *C.st_config {
	return &C.st_config{
		refined_length_blocks:               C.int(c.Filter.Refined.LengthBlocks),
		refined_initial_length_blocks:       C.int(c.Filter.RefinedInitial.LengthBlocks),
		use_linear_filter:                   stB(c.Filter.UseLinearFilter),
		conservative_initial_phase:          stB(c.Filter.ConservativeInitialPhase),
		initial_state_seconds:               C.float(c.Filter.InitialStateSeconds),
		erle_min:                            C.float(c.Erle.Min),
		erle_max_l:                          C.float(c.Erle.MaxL),
		erle_max_h:                          C.float(c.Erle.MaxH),
		erle_onset_detection:                stB(c.Erle.OnsetDetection),
		erle_num_sections:                   C.int(c.Erle.NumSections),
		erle_clamp_quality_estimate_to_zero: stB(c.Erle.ClampQualityEstimateToZero),
		erle_clamp_quality_estimate_to_one:  stB(c.Erle.ClampQualityEstimateToOne),
		default_gain:                        C.float(c.EpStrength.DefaultGain),
		default_len:                         C.float(c.EpStrength.DefaultLen),
		nearend_len:                         C.float(c.EpStrength.NearendLen),
		echo_can_saturate:                   stB(c.EpStrength.EchoCanSaturate),
		bounded_erl:                         stB(c.EpStrength.BoundedErl),
		erle_onset_compensation_in_dominant_nearend: stB(c.EpStrength.ErleOnsetCompensationInDominantNearend),
		use_conservative_tail_frequency_response:    stB(c.EpStrength.UseConservativeTailFrequencyResponse),
		use_stationarity_properties:                 stB(c.EchoAudibility.UseStationarityProperties),
		use_stationarity_properties_at_init:         stB(c.EchoAudibility.UseStationarityPropertiesAtInit),
		active_render_limit:                         C.float(c.RenderLevels.ActiveRenderLimit),
		delay_headroom_samples:                      C.int(c.Delay.DelayHeadroomSamples),
		linear_and_stable_echo_path:                 stB(c.EchoRemovalControl.LinearAndStableEchoPath),
		noise_floor_hold:                            C.int(c.EchoModel.NoiseFloorHold),
		min_noise_floor_power:                       C.float(c.EchoModel.MinNoiseFloorPower),
		stationary_gate_slope:                       C.float(c.EchoModel.StationaryGateSlope),
		noise_gate_power:                            C.float(c.EchoModel.NoiseGatePower),
		noise_gate_slope:                            C.float(c.EchoModel.NoiseGateSlope),
		render_pre_window_size:                      C.int(c.EchoModel.RenderPreWindowSize),
		render_post_window_size:                     C.int(c.EchoModel.RenderPostWindowSize),
		model_reverb_in_nonlinear_mode:              stB(c.EchoModel.ModelReverbInNonlinearMode),
	}
}

// StRender pairs a Go RenderBuffer with an identical C++ one.
type StRender struct {
	c        unsafe.Pointer
	Blocks   *aec3.BlockBuffer
	Spectra  *aec3.SpectrumBuffer
	Ffts     *aec3.FftBuffer
	Buffer   *aec3.RenderBuffer
	Size     int
	Channels int
}

func NewStRender(size, numChannels int) *StRender {
	r := &StRender{
		c:        C.st_render_new(C.int(size), C.int(numChannels)),
		Blocks:   aec3.NewBlockBuffer(size, 1, numChannels),
		Spectra:  aec3.NewSpectrumBuffer(size, numChannels),
		Ffts:     aec3.NewFftBuffer(size, numChannels),
		Size:     size,
		Channels: numChannels,
	}
	r.Buffer = aec3.NewRenderBuffer(r.Blocks, r.Spectra, r.Ffts)
	return r
}

func (r *StRender) Free() { C.st_render_free(r.c) }

// SetBlock writes band 0 of channel ch in slot.
func (r *StRender) SetBlock(slot, ch int, x []float32) {
	copy(r.Blocks.At(slot).View(0, ch), x)
	C.st_render_set_block(r.c, C.int(slot), C.int(ch), stFs(x))
}

func (r *StRender) SetSpectrum(slot, ch int, x2 *[aec3.FftLengthBy2Plus1]float32) {
	r.Spectra.At(slot)[ch] = *x2
	C.st_render_set_spectrum(r.c, C.int(slot), C.int(ch), stF(&x2[0]))
}

func (r *StRender) SetIndices(write, read int) {
	r.Blocks.Write, r.Spectra.Write, r.Ffts.Write = write, write, write
	r.Blocks.Read, r.Spectra.Read, r.Ffts.Read = read, read, read
	C.st_render_set_indices(r.c, C.int(write), C.int(read))
}

// StMovingAverage wraps aec3::MovingAverage.
type StMovingAverage struct{ c unsafe.Pointer }

func NewStMovingAverage(numElem, memLen int) *StMovingAverage {
	return &StMovingAverage{C.st_moving_average_new(C.int(numElem), C.int(memLen))}
}
func (m *StMovingAverage) Free() { C.st_moving_average_free(m.c) }
func (m *StMovingAverage) Average(in, out []float32) {
	C.st_moving_average_average(m.c, stFs(in), stFs(out), C.int(len(in)))
}

// StReverbModel wraps ReverbModel.
type StReverbModel struct{ c unsafe.Pointer }

func NewStReverbModel() *StReverbModel { return &StReverbModel{C.st_reverb_model_new()} }
func (m *StReverbModel) Free()         { C.st_reverb_model_free(m.c) }
func (m *StReverbModel) UpdateReverbNoFreqShaping(ps []float32, scaling, decay float32) {
	C.st_reverb_model_update_no_freq_shaping(m.c, stFs(ps), C.int(len(ps)), C.float(scaling), C.float(decay))
}
func (m *StReverbModel) UpdateReverb(ps, scaling []float32, decay float32) {
	C.st_reverb_model_update(m.c, stFs(ps), stFs(scaling), C.int(len(ps)), C.float(decay))
}
func (m *StReverbModel) Reverb() (out [aec3.FftLengthBy2Plus1]float32) {
	C.st_reverb_model_reverb(m.c, stF(&out[0]))
	return
}

// StReverbFrequencyResponse wraps ReverbFrequencyResponse.
type StReverbFrequencyResponse struct{ c unsafe.Pointer }

func NewStReverbFrequencyResponse(conservative bool) *StReverbFrequencyResponse {
	return &StReverbFrequencyResponse{C.st_reverb_frequency_response_new(stB(conservative))}
}
func (m *StReverbFrequencyResponse) Free() { C.st_reverb_frequency_response_free(m.c) }
func (m *StReverbFrequencyResponse) Update(fr [][aec3.FftLengthBy2Plus1]float32, delay int, q aec3.LinearFilterQuality, stationary bool) {
	C.st_reverb_frequency_response_update(m.c, stSpec(fr), C.int(len(fr)), C.int(delay), stB(q.Ok), C.float(q.Value), stB(stationary))
}
func (m *StReverbFrequencyResponse) FrequencyResponse() (out [aec3.FftLengthBy2Plus1]float32) {
	C.st_reverb_frequency_response_get(m.c, stF(&out[0]))
	return
}

// StReverbDecayEstimator wraps ReverbDecayEstimator.
type StReverbDecayEstimator struct{ c unsafe.Pointer }

func NewStReverbDecayEstimator(cfg *aec3.EchoCanceller3Config) *StReverbDecayEstimator {
	return &StReverbDecayEstimator{C.st_reverb_decay_new(stCConfig(cfg))}
}
func (m *StReverbDecayEstimator) Free() { C.st_reverb_decay_free(m.c) }
func (m *StReverbDecayEstimator) Update(filter []float32, q aec3.LinearFilterQuality, delay int, usable, stationary bool) {
	C.st_reverb_decay_update(m.c, stFs(filter), C.int(len(filter)), stB(q.Ok), C.float(q.Value), C.int(delay), stB(usable), stB(stationary))
}
func (m *StReverbDecayEstimator) Decay(mild bool) float32 {
	return float32(C.st_reverb_decay_decay(m.c, stB(mild)))
}

// StErlEstimator wraps ErlEstimator.
type StErlEstimator struct{ c unsafe.Pointer }

func NewStErlEstimator(startup int) *StErlEstimator {
	return &StErlEstimator{C.st_erl_new(C.int(startup))}
}
func (m *StErlEstimator) Free()  { C.st_erl_free(m.c) }
func (m *StErlEstimator) Reset() { C.st_erl_reset(m.c) }
func (m *StErlEstimator) Update(converged []bool, render, capture [][aec3.FftLengthBy2Plus1]float32) {
	C.st_erl_update(m.c, stInts(stBools(converged)), C.int(len(converged)), stSpec(render), C.int(len(render)), stSpec(capture))
}
func (m *StErlEstimator) Get() (erl [aec3.FftLengthBy2Plus1]float32, td float32) {
	C.st_erl_get(m.c, stF(&erl[0]), stF(&td))
	return
}

func stQualities(f func(q *C.float, ok *C.int), n int) []aec3.LinearFilterQuality {
	q := make([]float32, n)
	ok := make([]C.int, n)
	f(stFs(q), stInts(ok))
	out := make([]aec3.LinearFilterQuality, n)
	for i := range out {
		out[i] = aec3.LinearFilterQuality{Value: q[i], Ok: ok[i] != 0}
	}
	return out
}

// StFullBandErleEstimator wraps FullBandErleEstimator.
type StFullBandErleEstimator struct {
	c   unsafe.Pointer
	nch int
}

func NewStFullBandErleEstimator(cfg *aec3.EchoCanceller3Config, nch int) *StFullBandErleEstimator {
	return &StFullBandErleEstimator{C.st_fullband_erle_new(stCConfig(cfg), C.int(nch)), nch}
}
func (m *StFullBandErleEstimator) Free()  { C.st_fullband_erle_free(m.c) }
func (m *StFullBandErleEstimator) Reset() { C.st_fullband_erle_reset(m.c) }
func (m *StFullBandErleEstimator) Update(X2 *[aec3.FftLengthBy2Plus1]float32, Y2, E2 [][aec3.FftLengthBy2Plus1]float32, converged []bool) {
	C.st_fullband_erle_update(m.c, stF(&X2[0]), stSpec(Y2), stSpec(E2), stInts(stBools(converged)))
}
func (m *StFullBandErleEstimator) FullbandErleLog2() float32 {
	return float32(C.st_fullband_erle_log2(m.c))
}
func (m *StFullBandErleEstimator) Qualities() []aec3.LinearFilterQuality {
	return stQualities(func(q *C.float, ok *C.int) { C.st_fullband_erle_qualities(m.c, q, ok) }, m.nch)
}

// StSubbandErleEstimator wraps SubbandErleEstimator.
type StSubbandErleEstimator struct {
	c   unsafe.Pointer
	nch int
}

func NewStSubbandErleEstimator(cfg *aec3.EchoCanceller3Config, nch int) *StSubbandErleEstimator {
	return &StSubbandErleEstimator{C.st_subband_erle_new(stCConfig(cfg), C.int(nch)), nch}
}
func (m *StSubbandErleEstimator) Free()  { C.st_subband_erle_free(m.c) }
func (m *StSubbandErleEstimator) Reset() { C.st_subband_erle_reset(m.c) }
func (m *StSubbandErleEstimator) Update(X2 *[aec3.FftLengthBy2Plus1]float32, Y2, E2 [][aec3.FftLengthBy2Plus1]float32, converged []bool) {
	C.st_subband_erle_update(m.c, stF(&X2[0]), stSpec(Y2), stSpec(E2), stInts(stBools(converged)))
}

// Get returns the estimate which selects: 0 erle, 1 onset compensated,
// 2 unbounded, 3 during onsets.
func (m *StSubbandErleEstimator) Get(which int) [][aec3.FftLengthBy2Plus1]float32 {
	out := make([][aec3.FftLengthBy2Plus1]float32, m.nch)
	C.st_subband_erle_get(m.c, C.int(which), stSpec(out))
	return out
}

// StSignalDependentErleEstimator wraps SignalDependentErleEstimator.
type StSignalDependentErleEstimator struct {
	c   unsafe.Pointer
	nch int
}

func NewStSignalDependentErleEstimator(cfg *aec3.EchoCanceller3Config, nch int) *StSignalDependentErleEstimator {
	return &StSignalDependentErleEstimator{C.st_sd_erle_new(stCConfig(cfg), C.int(nch)), nch}
}
func (m *StSignalDependentErleEstimator) Free()  { C.st_sd_erle_free(m.c) }
func (m *StSignalDependentErleEstimator) Reset() { C.st_sd_erle_reset(m.c) }
func (m *StSignalDependentErleEstimator) Update(r *StRender, fr [][][aec3.FftLengthBy2Plus1]float32,
	X2 *[aec3.FftLengthBy2Plus1]float32, Y2, E2, avgErle, avgErleOc [][aec3.FftLengthBy2Plus1]float32, converged []bool) {
	flat := stFlatFreqResp(fr)
	C.st_sd_erle_update(m.c, r.c, stFs(flat), C.int(len(fr[0])), stF(&X2[0]), stSpec(Y2), stSpec(E2),
		stSpec(avgErle), stSpec(avgErleOc), stInts(stBools(converged)))
}
func (m *StSignalDependentErleEstimator) Erle(onsetCompensated bool) [][aec3.FftLengthBy2Plus1]float32 {
	out := make([][aec3.FftLengthBy2Plus1]float32, m.nch)
	C.st_sd_erle_get(m.c, stB(onsetCompensated), stSpec(out))
	return out
}

// StErleEstimator wraps ErleEstimator.
type StErleEstimator struct {
	c   unsafe.Pointer
	nch int
}

func NewStErleEstimator(cfg *aec3.EchoCanceller3Config, startup, nch int) *StErleEstimator {
	return &StErleEstimator{C.st_erle_new(stCConfig(cfg), C.int(startup), C.int(nch)), nch}
}
func (m *StErleEstimator) Free()                  { C.st_erle_free(m.c) }
func (m *StErleEstimator) Reset(delayChange bool) { C.st_erle_reset(m.c, stB(delayChange)) }
func (m *StErleEstimator) Update(r *StRender, fr [][][aec3.FftLengthBy2Plus1]float32, X2 *[aec3.FftLengthBy2Plus1]float32,
	Y2, E2 [][aec3.FftLengthBy2Plus1]float32, converged []bool) {
	flat := stFlatFreqResp(fr)
	C.st_erle_update(m.c, r.c, stFs(flat), C.int(len(fr[0])), stF(&X2[0]), stSpec(Y2), stSpec(E2), stInts(stBools(converged)))
}

// Get returns the estimate which selects: 0 erle, 1 onset compensated,
// 2 unbounded, 3 during onsets.
func (m *StErleEstimator) Get(which int) [][aec3.FftLengthBy2Plus1]float32 {
	out := make([][aec3.FftLengthBy2Plus1]float32, m.nch)
	C.st_erle_get(m.c, C.int(which), stSpec(out))
	return out
}
func (m *StErleEstimator) FullbandErleLog2() float32 { return float32(C.st_erle_fullband_log2(m.c)) }
func (m *StErleEstimator) Qualities() []aec3.LinearFilterQuality {
	return stQualities(func(q *C.float, ok *C.int) { C.st_erle_qualities(m.c, q, ok) }, m.nch)
}

// StStationarityEstimator wraps StationarityEstimator.
type StStationarityEstimator struct{ c unsafe.Pointer }

func NewStStationarityEstimator() *StStationarityEstimator {
	return &StStationarityEstimator{C.st_stationarity_new()}
}
func (m *StStationarityEstimator) Free() { C.st_stationarity_free(m.c) }
func (m *StStationarityEstimator) UpdateNoiseEstimator(s [][aec3.FftLengthBy2Plus1]float32) {
	C.st_stationarity_update_noise(m.c, stSpec(s), C.int(len(s)))
}
func (m *StStationarityEstimator) UpdateStationarityFlags(r *StRender, reverb *[aec3.FftLengthBy2Plus1]float32, idxCurrent, numLookahead int) {
	C.st_stationarity_update_flags(m.c, r.c, stF(&reverb[0]), C.int(idxCurrent), C.int(numLookahead))
}
func (m *StStationarityEstimator) Flags() (bands [aec3.FftLengthBy2Plus1]bool, block bool) {
	var b [aec3.FftLengthBy2Plus1]C.int
	var blk C.int
	C.st_stationarity_flags(m.c, &b[0], &blk)
	for i := range b {
		bands[i] = b[i] != 0
	}
	return bands, blk != 0
}

// StEchoAudibility wraps EchoAudibility.
type StEchoAudibility struct{ c unsafe.Pointer }

func NewStEchoAudibility(atInit bool) *StEchoAudibility {
	return &StEchoAudibility{C.st_echo_audibility_new(stB(atInit))}
}
func (m *StEchoAudibility) Free() { C.st_echo_audibility_free(m.c) }
func (m *StEchoAudibility) Update(r *StRender, reverb *[aec3.FftLengthBy2Plus1]float32, delay int, externalDelaySeen bool) {
	C.st_echo_audibility_update(m.c, r.c, stF(&reverb[0]), C.int(delay), stB(externalDelaySeen))
}
func (m *StEchoAudibility) Get(converged bool) (scaling [aec3.FftLengthBy2Plus1]float32, blockStationary bool) {
	var b C.int
	C.st_echo_audibility_get(m.c, stB(converged), stF(&scaling[0]), &b)
	return scaling, b != 0
}

// StFilterAnalyzer wraps FilterAnalyzer.
type StFilterAnalyzer struct {
	c   unsafe.Pointer
	nch int
}

func NewStFilterAnalyzer(cfg *aec3.EchoCanceller3Config, nch int) *StFilterAnalyzer {
	return &StFilterAnalyzer{C.st_filter_analyzer_new(stCConfig(cfg), C.int(nch)), nch}
}
func (m *StFilterAnalyzer) Free()  { C.st_filter_analyzer_free(m.c) }
func (m *StFilterAnalyzer) Reset() { C.st_filter_analyzer_reset(m.c) }
func (m *StFilterAnalyzer) Update(filters [][]float32, r *StRender) (bool, float32) {
	flat := stFlat(filters)
	var consistent C.int
	var gain float32
	C.st_filter_analyzer_update(m.c, stFs(flat), C.int(len(filters[0])), r.c, &consistent, stF(&gain))
	return consistent != 0, gain
}
func (m *StFilterAnalyzer) Get(filterLen int) (delays []int, minDelay, lengthBlocks int, adjusted [][]float32) {
	d := make([]C.int, m.nch)
	flat := make([]float32, m.nch*filterLen)
	var md, lb C.int
	C.st_filter_analyzer_get(m.c, stInts(d), &md, &lb, stFs(flat))
	delays = make([]int, m.nch)
	adjusted = make([][]float32, m.nch)
	for ch := range delays {
		delays[ch] = int(d[ch])
		adjusted[ch] = flat[ch*filterLen : (ch+1)*filterLen]
	}
	return delays, int(md), int(lb), adjusted
}

// StSubtractorOutputAnalyzer wraps SubtractorOutputAnalyzer.
type StSubtractorOutputAnalyzer struct {
	c   unsafe.Pointer
	nch int
}

func NewStSubtractorOutputAnalyzer(nch int) *StSubtractorOutputAnalyzer {
	return &StSubtractorOutputAnalyzer{C.st_subtractor_analyzer_new(C.int(nch)), nch}
}
func (m *StSubtractorOutputAnalyzer) Free() { C.st_subtractor_analyzer_free(m.c) }
func (m *StSubtractorOutputAnalyzer) Update(so []aec3.SubtractorOutput) (out [3]bool, converged []bool) {
	y2 := make([]float32, m.nch)
	e2r := make([]float32, m.nch)
	e2c := make([]float32, m.nch)
	for ch := range so {
		y2[ch], e2r[ch], e2c[ch] = so[ch].Y2, so[ch].E2RefinedSum, so[ch].E2CoarseSum
	}
	var o [3]C.int
	conv := make([]C.int, m.nch)
	C.st_subtractor_analyzer_update(m.c, stFs(y2), stFs(e2r), stFs(e2c), &o[0], stInts(conv))
	converged = make([]bool, m.nch)
	for ch := range conv {
		converged[ch] = conv[ch] != 0
	}
	return [3]bool{o[0] != 0, o[1] != 0, o[2] != 0}, converged
}
func (m *StSubtractorOutputAnalyzer) HandleEchoPathChange() {
	C.st_subtractor_analyzer_handle_echo_path_change(m.c)
}

// StTransparentMode wraps the classifier TransparentMode::Create returns.
type StTransparentMode struct{ c unsafe.Pointer }

// NewStTransparentMode returns nil when TransparentMode::Create returns no
// classifier; hmm enables the WebRTC-Aec3TransparentModeHmm field trial.
func NewStTransparentMode(cfg *aec3.EchoCanceller3Config, hmm bool) *StTransparentMode {
	p := C.st_transparent_mode_new(stCConfig(cfg), stB(hmm))
	if p == nil {
		return nil
	}
	return &StTransparentMode{p}
}
func (m *StTransparentMode) Free()  { C.st_transparent_mode_free(m.c) }
func (m *StTransparentMode) Reset() { C.st_transparent_mode_reset(m.c) }
func (m *StTransparentMode) Update(delay int, consistent, converged, coarseConverged, diverged, activeRender, saturatedCapture bool) bool {
	return C.st_transparent_mode_update(m.c, C.int(delay), stB(consistent), stB(converged), stB(coarseConverged),
		stB(diverged), stB(activeRender), stB(saturatedCapture)) != 0
}

// StAecStateOut holds every AecState getter.
type StAecStateOut struct {
	Erle, ErleOc, ErleUnbounded                 [][aec3.FftLengthBy2Plus1]float32
	Erl, ResidualScaling, ReverbFreqResp        [aec3.FftLengthBy2Plus1]float32
	ErlTimeDomain, FullbandErleLog2             float32
	ReverbDecay, ReverbDecayMild                float32
	UsableLinearEstimate, UseLinearFilterOutput bool
	ActiveRender, UseStationarityProperties     bool
	MinDirectPathFilterDelay                    int
	SaturatedCapture, SaturatedEcho             bool
	TransparentModeActive, TransitionTriggered  bool
	FilterLengthBlocks                          int
}

// StAecState wraps AecState.
type StAecState struct {
	c   unsafe.Pointer
	nch int
}

func NewStAecState(cfg *aec3.EchoCanceller3Config, nch int, hmm bool) *StAecState {
	return &StAecState{C.st_aec_state_new(stCConfig(cfg), C.int(nch), stB(hmm)), nch}
}
func (m *StAecState) Free() { C.st_aec_state_free(m.c) }
func (m *StAecState) HandleEchoPathChange(v aec3.EchoPathVariability) {
	C.st_aec_state_handle_echo_path_change(m.c, stB(v.GainChange), C.int(v.DelayChange))
}
func (m *StAecState) UpdateCaptureSaturation(s bool) {
	C.st_aec_state_update_capture_saturation(m.c, stB(s))
}
func (m *StAecState) Update(ext *aec3.DelayEstimate, fr [][][aec3.FftLengthBy2Plus1]float32, ir [][]float32, r *StRender,
	E2, Y2 [][aec3.FftLengthBy2Plus1]float32, so []aec3.SubtractorOutput) {
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
	C.st_aec_state_update(m.c, extValid, extQuality, extDelay, stFs(flatFr), C.int(len(fr[0])), stFs(flatIr),
		C.int(len(ir[0])), r.c, stSpec(E2), stSpec(Y2), stFs(sub))
}
func (m *StAecState) Get() StAecStateOut {
	var o C.st_aec_state_out
	C.st_aec_state_get(m.c, &o)
	out := StAecStateOut{
		Erle:                      make([][aec3.FftLengthBy2Plus1]float32, m.nch),
		ErleOc:                    make([][aec3.FftLengthBy2Plus1]float32, m.nch),
		ErleUnbounded:             make([][aec3.FftLengthBy2Plus1]float32, m.nch),
		ErlTimeDomain:             float32(o.erl_time_domain),
		FullbandErleLog2:          float32(o.fullband_erle_log2),
		ReverbDecay:               float32(o.reverb_decay),
		ReverbDecayMild:           float32(o.reverb_decay_mild),
		UsableLinearEstimate:      o.usable_linear_estimate != 0,
		UseLinearFilterOutput:     o.use_linear_filter_output != 0,
		ActiveRender:              o.active_render != 0,
		UseStationarityProperties: o.use_stationarity_properties != 0,
		MinDirectPathFilterDelay:  int(o.min_direct_path_filter_delay),
		SaturatedCapture:          o.saturated_capture != 0,
		SaturatedEcho:             o.saturated_echo != 0,
		TransparentModeActive:     o.transparent_mode_active != 0,
		TransitionTriggered:       o.transition_triggered != 0,
		FilterLengthBlocks:        int(o.filter_length_blocks),
	}
	for ch := 0; ch < m.nch; ch++ {
		for k := 0; k < aec3.FftLengthBy2Plus1; k++ {
			out.Erle[ch][k] = float32(o.erle[ch][k])
			out.ErleOc[ch][k] = float32(o.erle_oc[ch][k])
			out.ErleUnbounded[ch][k] = float32(o.erle_unbounded[ch][k])
		}
	}
	for k := 0; k < aec3.FftLengthBy2Plus1; k++ {
		out.Erl[k] = float32(o.erl[k])
		out.ResidualScaling[k] = float32(o.residual_scaling[k])
		out.ReverbFreqResp[k] = float32(o.reverb_freq_resp[k])
	}
	return out
}

// StResidualEchoEstimator wraps ResidualEchoEstimator.
type StResidualEchoEstimator struct{ c unsafe.Pointer }

func NewStResidualEchoEstimator(cfg *aec3.EchoCanceller3Config, numRender int) *StResidualEchoEstimator {
	return &StResidualEchoEstimator{C.st_residual_echo_new(stCConfig(cfg), C.int(numRender))}
}
func (m *StResidualEchoEstimator) Free() { C.st_residual_echo_free(m.c) }
func (m *StResidualEchoEstimator) Estimate(a *StAecState, r *StRender, S2, Y2 [][aec3.FftLengthBy2Plus1]float32,
	dominantNearend bool, R2, R2u [][aec3.FftLengthBy2Plus1]float32) {
	C.st_residual_echo_estimate(m.c, a.c, r.c, stSpec(S2), stSpec(Y2), C.int(len(S2)), stB(dominantNearend),
		stSpec(R2), stSpec(R2u))
}
