//go:build webrtc_oracle

package webrtcoracle

/*
#include "erem_oracle.h"
*/
import "C"

import (
	"math"
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

// EremKind selects the C++ EchoRemover build an EremSim runs.
type EremKind int

const (
	// EremPrebuilt is webrtc::EchoRemover::Create from libwebrtc.a (lane =
	// DetectOptimization(), kNeon on arm64).
	EremPrebuilt EremKind = iota
	// EremSource is echo_remover.cc built from source on the prebuilt
	// dependencies with a forced lane.
	EremSource
	// EremScalarFft is the source build of echo_remover.cc and its FFT users
	// on the scalar Ooura FFT with a forced lane.
	EremScalarFft
)

// EremSim is a C++ EchoRemover with its multi-band RenderBuffer.
type EremSim struct {
	p                     *C.erem_sim
	numBands, nr, nc      int
	blocks, fft, spectrum []float32
	capture, linear       []float32
}

func eremBool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// eremFlattenConfig flattens c in the order of EREM_CONFIG in erem_shim.h.
func eremFlattenConfig(c *aec3.EchoCanceller3Config) ([]float32, []C.int) {
	var f []float32
	var i []C.int
	F := func(v float32) { f = append(f, v) }
	I := func(v int) { i = append(i, C.int(v)) }
	B := func(v bool) { i = append(i, eremBool(v)) }
	I(c.Buffering.ExcessRenderDetectionIntervalBlocks)
	I(c.Buffering.MaxAllowedExcessRenderBlocks)
	d := &c.Delay
	I(d.DefaultDelay)
	I(d.DownSamplingFactor)
	I(d.NumFilters)
	I(d.DelayHeadroomSamples)
	I(d.HysteresisLimitBlocks)
	I(d.FixedCaptureDelaySamples)
	F(d.DelayEstimateSmoothing)
	F(d.DelayEstimateSmoothingDelayFound)
	F(d.DelayCandidateDetectionThreshold)
	I(d.DelaySelectionThresholds.Initial)
	I(d.DelaySelectionThresholds.Converged)
	B(d.UseExternalDelayEstimator)
	B(d.LogWarningOnDelayChanges)
	for _, m := range []aec3.AlignmentMixing{d.RenderAlignmentMixing, d.CaptureAlignmentMixing} {
		B(m.Downmix)
		B(m.AdaptiveSelection)
		F(m.ActivityPowerThreshold)
		B(m.PreferFirstTwoChannels)
	}
	B(d.DetectPreEcho)
	fl := &c.Filter
	refined := func(r aec3.RefinedConfiguration) {
		I(r.LengthBlocks)
		F(r.LeakageConverged)
		F(r.LeakageDiverged)
		F(r.ErrorFloor)
		F(r.ErrorCeil)
		F(r.NoiseGate)
	}
	coarse := func(r aec3.CoarseConfiguration) {
		I(r.LengthBlocks)
		F(r.Rate)
		F(r.NoiseGate)
	}
	refined(fl.Refined)
	coarse(fl.Coarse)
	refined(fl.RefinedInitial)
	coarse(fl.CoarseInitial)
	I(fl.ConfigChangeDurationBlocks)
	F(fl.InitialStateSeconds)
	I(fl.CoarseResetHangoverBlocks)
	B(fl.ConservativeInitialPhase)
	B(fl.EnableCoarseFilterOutputUsage)
	B(fl.UseLinearFilter)
	B(fl.HighPassFilterEchoReference)
	B(fl.ExportLinearAecOutput)
	F(c.Erle.Min)
	F(c.Erle.MaxL)
	F(c.Erle.MaxH)
	B(c.Erle.OnsetDetection)
	I(c.Erle.NumSections)
	B(c.Erle.ClampQualityEstimateToZero)
	B(c.Erle.ClampQualityEstimateToOne)
	ep := &c.EpStrength
	F(ep.DefaultGain)
	F(ep.DefaultLen)
	F(ep.NearendLen)
	B(ep.EchoCanSaturate)
	B(ep.BoundedErl)
	B(ep.ErleOnsetCompensationInDominantNearend)
	B(ep.UseConservativeTailFrequencyResponse)
	ea := &c.EchoAudibility
	F(ea.LowRenderLimit)
	F(ea.NormalRenderLimit)
	F(ea.FloorPower)
	F(ea.AudibilityThresholdLf)
	F(ea.AudibilityThresholdMf)
	F(ea.AudibilityThresholdHf)
	B(ea.UseStationarityProperties)
	B(ea.UseStationarityPropertiesAtInit)
	rl := &c.RenderLevels
	F(rl.ActiveRenderLimit)
	F(rl.PoorExcitationRenderLimit)
	F(rl.PoorExcitationRenderLimitDs8)
	F(rl.RenderPowerGainDb)
	B(c.EchoRemovalControl.HasClockDrift)
	B(c.EchoRemovalControl.LinearAndStableEchoPath)
	em := &c.EchoModel
	I(em.NoiseFloorHold)
	F(em.MinNoiseFloorPower)
	F(em.StationaryGateSlope)
	F(em.NoiseGatePower)
	F(em.NoiseGateSlope)
	I(em.RenderPreWindowSize)
	I(em.RenderPostWindowSize)
	B(em.ModelReverbInNonlinearMode)
	F(c.ComfortNoise.NoiseFloorDbfs)
	s := &c.Suppressor
	I(s.NearendAverageBlocks)
	for _, t := range []aec3.Tuning{s.NormalTuning, s.NearendTuning} {
		for _, m := range []aec3.MaskingThresholds{t.MaskLf, t.MaskHf} {
			F(m.EnrTransparent)
			F(m.EnrSuppress)
			F(m.EmrTransparent)
		}
		F(t.MaxIncFactor)
		F(t.MaxDecFactorLf)
	}
	B(s.LfSmoothingDuringInitialPhase)
	I(s.LastPermanentLfSmoothingBand)
	I(s.LastLfSmoothingBand)
	I(s.LastLfBand)
	I(s.FirstHfBand)
	dn := &s.DominantNearendDetection
	F(dn.EnrThreshold)
	F(dn.EnrExitThreshold)
	F(dn.SnrThreshold)
	I(dn.HoldDuration)
	I(dn.TriggerThreshold)
	B(dn.UseDuringInitialPhase)
	B(dn.UseUnboundedEchoSpectrum)
	sn := &s.SubbandNearendDetection
	I(sn.NearendAverageBlocks)
	I(sn.Subband1.Low)
	I(sn.Subband1.High)
	I(sn.Subband2.Low)
	I(sn.Subband2.High)
	F(sn.NearendThreshold)
	F(sn.SnrThreshold)
	B(s.UseSubbandNearendDetection)
	hb := &s.HighBandsSuppression
	F(hb.EnrThreshold)
	F(hb.MaxGainDuringEcho)
	F(hb.AntiHowlingActivationThreshold)
	F(hb.AntiHowlingGain)
	F(s.FloorFirstIncrease)
	B(s.ConservativeHfSuppression)
	mc := &c.MultiChannel
	B(mc.DetectStereoContent)
	F(mc.StereoDetectionThreshold)
	I(mc.StereoDetectionTimeoutThresholdSeconds)
	F(mc.StereoDetectionHysteresisSeconds)
	return f, i
}

// NewEremSim creates the C++ EchoRemover of the given kind. neon selects
// kNeon for the source builds (ignored for EremPrebuilt).
func NewEremSim(cfg *aec3.EchoCanceller3Config, rate, numRender, numCapture, renderSize int, kind EremKind,
	neon bool) *EremSim {
	f, i := eremFlattenConfig(cfg)
	nb := aec3.NumBandsForRate(rate)
	s := &EremSim{numBands: nb, nr: numRender, nc: numCapture,
		blocks:   make([]float32, nb*numRender*aec3.BlockSize),
		fft:      make([]float32, numRender*2*aec3.FftLengthBy2Plus1),
		spectrum: make([]float32, numRender*aec3.FftLengthBy2Plus1),
		capture:  make([]float32, nb*numCapture*aec3.BlockSize),
		linear:   make([]float32, numCapture*aec3.BlockSize),
	}
	s.p = C.erem_new((*C.float)(unsafe.Pointer(&f[0])), &i[0], C.int(rate), C.int(numRender), C.int(numCapture),
		C.int(renderSize), C.int(kind), eremBool(neon))
	return s
}

func (s *EremSim) Free() { C.erem_free(s.p) }

// SetRender writes ring slot `slot` from the Go render rings.
func (s *EremSim) SetRender(slot int, block *aec3.Block, X []aec3.FftData, X2 [][aec3.FftLengthBy2Plus1]float32) {
	for b := 0; b < s.numBands; b++ {
		for ch := 0; ch < s.nr; ch++ {
			copy(s.blocks[(b*s.nr+ch)*aec3.BlockSize:], block.View(b, ch))
		}
	}
	for ch := 0; ch < s.nr; ch++ {
		copy(s.fft[ch*130:], X[ch].Re[:])
		copy(s.fft[ch*130+65:], X[ch].Im[:])
		copy(s.spectrum[ch*65:], X2[ch][:])
	}
	C.erem_set_render(s.p, C.int(slot), (*C.float)(unsafe.Pointer(&s.blocks[0])),
		(*C.float)(unsafe.Pointer(&s.fft[0])), (*C.float)(unsafe.Pointer(&s.spectrum[0])))
}

func (s *EremSim) SetIndices(write, read int) { C.erem_set_indices(s.p, C.int(write), C.int(read)) }

// ProcessCapture runs EchoRemover::ProcessCapture on capture in place; when
// linear is non-nil it receives the linear output.
func (s *EremSim) ProcessCapture(v aec3.EchoPathVariability, saturation bool, ext *aec3.DelayEstimate,
	linear, capture *aec3.Block) {
	for b := 0; b < s.numBands; b++ {
		for ch := 0; ch < s.nc; ch++ {
			copy(s.capture[(b*s.nc+ch)*aec3.BlockSize:], capture.View(b, ch))
		}
	}
	var ed [4]C.int
	var edp *C.int
	if ext != nil {
		ed = [4]C.int{C.int(ext.Quality), C.int(ext.Delay), C.int(ext.BlocksSinceLastChange),
			C.int(ext.BlocksSinceLastUpdate)}
		edp = &ed[0]
	}
	var lp *C.float
	if linear != nil {
		lp = (*C.float)(unsafe.Pointer(&s.linear[0]))
	}
	C.erem_process(s.p, eremBool(v.GainChange), C.int(v.DelayChange), eremBool(v.ClockDrift), eremBool(saturation),
		edp, (*C.float)(unsafe.Pointer(&s.capture[0])), lp)
	for b := 0; b < s.numBands; b++ {
		for ch := 0; ch < s.nc; ch++ {
			copy(capture.View(b, ch), s.capture[(b*s.nc+ch)*aec3.BlockSize:])
		}
	}
	if linear != nil {
		for ch := 0; ch < s.nc; ch++ {
			copy(linear.View(0, ch), s.linear[ch*aec3.BlockSize:])
		}
	}
}

func (s *EremSim) UpdateEchoLeakageStatus(l bool) { C.erem_set_leakage(s.p, eremBool(l)) }
func (s *EremSim) SetCaptureOutputUsage(u bool)   { C.erem_set_output_usage(s.p, eremBool(u)) }

// GetMetrics returns {echo_return_loss, echo_return_loss_enhancement}.
func (s *EremSim) GetMetrics() (float64, float64) {
	var out [2]C.double
	C.erem_get_metrics(s.p, &out[0])
	return float64(out[0]), float64(out[1])
}

// EremLog10f calls the host log10f through std::log10(float).
func EremLog10f(x float32) float32 { return float32(C.erem_log10f(C.float(x))) }

// EremTransformDbMetricForReporting calls the prebuilt
// aec3::TransformDbMetricForReporting.
func EremTransformDbMetricForReporting(negate bool, minV, maxV, offset, scaling, value float32) int {
	return int(C.erem_transform_db_metric(eremBool(negate), C.float(minV), C.float(maxV), C.float(offset),
		C.float(scaling), C.float(value)))
}

// goaec3EremLog10f backs the log10f interposer in erem_libm.c. It is the
// formula of aec3's erLog10f: the double log10 rounded to float.
//
//export goaec3EremLog10f
func goaec3EremLog10f(x C.float) C.float { return C.float(math.Log10(float64(x))) }
