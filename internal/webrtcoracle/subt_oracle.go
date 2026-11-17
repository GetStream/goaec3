//go:build webrtc_oracle

package webrtcoracle

/*
#include "subt_oracle.h"
*/
import "C"

import (
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

// Layout checks: subt_output mirrors aec3.SubtractorOutput.
var _ [unsafe.Sizeof(C.subt_output{}) - unsafe.Sizeof(aec3.SubtractorOutput{})]struct{}
var _ [unsafe.Sizeof(aec3.SubtractorOutput{}) - unsafe.Sizeof(C.subt_output{})]struct{}

func subtF(p *float32) *C.float { return (*C.float)(unsafe.Pointer(p)) }

func subtB(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func subtRefined(c aec3.RefinedConfiguration) [6]C.float {
	return [6]C.float{C.float(c.LengthBlocks), C.float(c.LeakageConverged), C.float(c.LeakageDiverged),
		C.float(c.ErrorFloor), C.float(c.ErrorCeil), C.float(c.NoiseGate)}
}

func subtCoarse(c aec3.CoarseConfiguration) [3]C.float {
	return [3]C.float{C.float(c.LengthBlocks), C.float(c.Rate), C.float(c.NoiseGate)}
}

// SubtAecState is the subset of the C++ AecState read back after each step.
type SubtAecState struct {
	MinDirectPathFilterDelay int
	TransitionTriggered      bool
	SaturatedCapture         bool
	UsableLinearEstimate     bool
	FullBandErleLog2         float32
}

// SubtSim wraps a C++ Subtractor driven with its own RenderBuffer,
// RenderSignalAnalyzer and AecState (see subt_oracle.h). Only the Filter
// section of the config is forwarded; every other field is the default.
type SubtSim struct {
	p   *C.subt_sim
	nch int
}

// NewSubtSim creates the C++ simulation. neon selects kNeon (else kNone);
// scalarFft selects the source build running the scalar Ooura FFT.
func NewSubtSim(cfg *aec3.EchoCanceller3Config, renderSize, numRender, numCapture int, neon, scalarFft bool) *SubtSim {
	f := &cfg.Filter
	c := C.subt_filter_config{
		refined:                           subtRefined(f.Refined),
		refined_initial:                   subtRefined(f.RefinedInitial),
		coarse:                            subtCoarse(f.Coarse),
		coarse_initial:                    subtCoarse(f.CoarseInitial),
		config_change_duration_blocks:     C.int(f.ConfigChangeDurationBlocks),
		initial_state_seconds:             C.float(f.InitialStateSeconds),
		coarse_reset_hangover_blocks:      C.int(f.CoarseResetHangoverBlocks),
		conservative_initial_phase:        subtB(f.ConservativeInitialPhase),
		enable_coarse_filter_output_usage: subtB(f.EnableCoarseFilterOutputUsage),
		use_linear_filter:                 subtB(f.UseLinearFilter),
	}
	return &SubtSim{C.subt_new(&c, C.int(renderSize), C.int(numRender), C.int(numCapture), subtB(neon), subtB(scalarFft)), numCapture}
}

func (s *SubtSim) Free() { C.subt_free(s.p) }

func (s *SubtSim) SetRender(slot, ch int, block []float32, X *aec3.FftData, X2 *[aec3.FftLengthBy2Plus1]float32) {
	C.subt_set_render(s.p, C.int(slot), C.int(ch), subtF(&block[0]), subtF(&X.Re[0]), subtF(&X2[0]))
}

func (s *SubtSim) SetIndices(write, read int) { C.subt_set_indices(s.p, C.int(write), C.int(read)) }

func (s *SubtSim) HandleEchoPathChange(v aec3.EchoPathVariability) {
	C.subt_handle_echo_path_change(s.p, subtB(v.GainChange), C.int(v.DelayChange))
}

func (s *SubtSim) ExitInitialState() { C.subt_exit_initial_state(s.p) }

// Process runs one step. capture is [nch*64], Y2 [nch][65]; out has nch
// entries.
func (s *SubtSim) Process(capture []float32, Y2 [][aec3.FftLengthBy2Plus1]float32, saturated, followTransition bool, out []aec3.SubtractorOutput) {
	C.subt_process(s.p, subtF(&capture[0]), subtF(&Y2[0][0]), subtB(saturated), subtB(followTransition),
		(*C.subt_output)(unsafe.Pointer(&out[0])))
}

func (s *SubtSim) AecState() SubtAecState {
	var o C.subt_aec_state
	C.subt_aec_state_get(s.p, &o)
	return SubtAecState{
		MinDirectPathFilterDelay: int(o.min_direct_path_filter_delay),
		TransitionTriggered:      o.transition_triggered != 0,
		SaturatedCapture:         o.saturated_capture != 0,
		UsableLinearEstimate:     o.usable_linear_estimate != 0,
		FullBandErleLog2:         float32(o.fullband_erle_log2),
	}
}

// FreqResp writes channel ch's refined frequency responses into H2 (capacity
// must cover the maximum filter length) and returns how many there are.
func (s *SubtSim) FreqResp(ch int, H2 [][aec3.FftLengthBy2Plus1]float32) int {
	return int(C.subt_freq_resp(s.p, C.int(ch), subtF(&H2[0][0])))
}

// ImpulseResp writes channel ch's refined impulse response into h and
// returns its length.
func (s *SubtSim) ImpulseResp(ch int, h []float32) int {
	return int(C.subt_impulse_resp(s.p, C.int(ch), subtF(&h[0])))
}
