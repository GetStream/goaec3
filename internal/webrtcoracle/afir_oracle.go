//go:build webrtc_oracle

package webrtcoracle

/*
#include "afir_oracle.h"
*/
import "C"

import (
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

// Lanes of the afir_* entry points.
const (
	AfirLaneScalar = 0
	AfirLaneNeon   = 1
)

func afirF(p *float32) *C.float { return (*C.float)(unsafe.Pointer(p)) }

// afirFftPtr passes d as 130 floats, re[65] then im[65], which is the Go
// layout of FftData.
func afirFftPtr(d *aec3.FftData) *C.float { return afirF(&d.Re[0]) }

// AfirRenderBuffer wraps a C++ RenderBuffer over its own Block/Spectrum/Fft buffers.
type AfirRenderBuffer struct{ p *C.afir_rb }

func NewAfirRenderBuffer(size, numBands, numChannels int) *AfirRenderBuffer {
	return &AfirRenderBuffer{C.afir_rb_new(C.int(size), C.int(numBands), C.int(numChannels))}
}
func (r *AfirRenderBuffer) Free() { C.afir_rb_free(r.p) }
func (r *AfirRenderBuffer) SetFft(index, ch int, d *aec3.FftData) {
	C.afir_rb_set_fft(r.p, C.int(index), C.int(ch), afirFftPtr(d))
}
func (r *AfirRenderBuffer) SetSpectrum(index, ch int, x *[aec3.FftLengthBy2Plus1]float32) {
	C.afir_rb_set_spectrum(r.p, C.int(index), C.int(ch), afirF(&x[0]))
}
func (r *AfirRenderBuffer) SetBlock(index, band, ch int, x []float32) {
	C.afir_rb_set_block(r.p, C.int(index), C.int(band), C.int(ch), afirF(&x[0]))
}
func (r *AfirRenderBuffer) SetRead(fftRead, spectrumRead, blockRead int) {
	C.afir_rb_set_read(r.p, C.int(fftRead), C.int(spectrumRead), C.int(blockRead))
}

// AfirFlatH packs H[partition][channel] contiguously.
func AfirFlatH(H aec3.FirCoefficients) []aec3.FftData {
	out := make([]aec3.FftData, 0, len(H.Flat()))
	out = append(out, H.Flat()...)
	return out
}

func AfirApplyFilter(rb *AfirRenderBuffer, lane, numPartitions int, H []aec3.FftData, hPartitions, numChannels int, S *aec3.FftData) {
	C.afir_apply_filter(rb.p, C.int(lane), C.int(numPartitions), afirFftPtr(&H[0]), C.int(hPartitions), C.int(numChannels), afirFftPtr(S))
}

func AfirAdaptPartitions(rb *AfirRenderBuffer, lane int, G *aec3.FftData, numPartitions int, H []aec3.FftData, hPartitions, numChannels int) {
	C.afir_adapt_partitions(rb.p, C.int(lane), afirFftPtr(G), C.int(numPartitions), afirFftPtr(&H[0]), C.int(hPartitions), C.int(numChannels))
}

func AfirFreqResp(lane, numPartitions int, H []aec3.FftData, hPartitions, numChannels int, H2 [][aec3.FftLengthBy2Plus1]float32) {
	C.afir_freq_resp(C.int(lane), C.int(numPartitions), afirFftPtr(&H[0]), C.int(hPartitions), C.int(numChannels), afirF(&H2[0][0]))
}

func AfirErl(lane int, H2 [][aec3.FftLengthBy2Plus1]float32, erl []float32) {
	var p *C.float
	if len(H2) > 0 {
		p = afirF(&H2[0][0])
	}
	C.afir_erl(C.int(lane), p, C.int(len(H2)), afirF(&erl[0]))
}

// AfirFilter wraps webrtc::AdaptiveFirFilter (scalarFft selects the
// source-built variant constrained through the scalar Ooura FFT).
type AfirFilter struct {
	p                  *C.afir_filter
	maxSize, nChannels int
}

func NewAfirFilter(scalarFft bool, maxSize, initialSize, changeDuration, numChannels, lane int) *AfirFilter {
	s := 0
	if scalarFft {
		s = 1
	}
	return &AfirFilter{C.afir_filter_new(C.int(s), C.int(maxSize), C.int(initialSize), C.int(changeDuration), C.int(numChannels), C.int(lane)), maxSize, numChannels}
}
func (f *AfirFilter) Free() { C.afir_filter_free(f.p) }
func (f *AfirFilter) Filter(rb *AfirRenderBuffer, S *aec3.FftData) {
	C.afir_filter_filter(f.p, rb.p, afirFftPtr(S))
}
func (f *AfirFilter) Adapt(rb *AfirRenderBuffer, G *aec3.FftData) {
	C.afir_filter_adapt(f.p, rb.p, afirFftPtr(G))
}

// AdaptWithImpulseResponse updates ir in place (cap must be the max length)
// and returns it resized.
func (f *AfirFilter) AdaptWithImpulseResponse(rb *AfirRenderBuffer, G *aec3.FftData, ir []float32) []float32 {
	full := ir[:cap(ir)]
	n := C.afir_filter_adapt_ir(f.p, rb.p, afirFftPtr(G), afirF(&full[0]), C.int(len(ir)))
	return full[:n]
}
func (f *AfirFilter) ComputeFrequencyResponse(H2 [][aec3.FftLengthBy2Plus1]float32) [][aec3.FftLengthBy2Plus1]float32 {
	full := H2[:cap(H2)]
	n := C.afir_filter_freq_resp(f.p, afirF(&full[0][0]))
	return full[:n]
}
func (f *AfirFilter) SetSizePartitions(size int, immediate bool) {
	i := 0
	if immediate {
		i = 1
	}
	C.afir_filter_set_size(f.p, C.int(size), C.int(i))
}
func (f *AfirFilter) HandleEchoPathChange()      { C.afir_filter_handle_echo_path_change(f.p) }
func (f *AfirFilter) ScaleFilter(factor float32) { C.afir_filter_scale(f.p, C.float(factor)) }
func (f *AfirFilter) SetFilter(numPartitions int, H []aec3.FftData) {
	C.afir_filter_set_filter(f.p, C.int(numPartitions), afirFftPtr(&H[0]))
}
func (f *AfirFilter) GetFilter(H []aec3.FftData) { C.afir_filter_get_filter(f.p, afirFftPtr(&H[0])) }
func (f *AfirFilter) SizePartitions() int        { return int(C.afir_filter_size(f.p)) }

// AfirRenderSignalAnalyzer wraps webrtc::RenderSignalAnalyzer.
type AfirRenderSignalAnalyzer struct{ p *C.afir_rsa }

func NewAfirRenderSignalAnalyzer(refinedLengthBlocks int) *AfirRenderSignalAnalyzer {
	return &AfirRenderSignalAnalyzer{C.afir_rsa_new(C.int(refinedLengthBlocks))}
}
func (a *AfirRenderSignalAnalyzer) Free() { C.afir_rsa_free(a.p) }
func (a *AfirRenderSignalAnalyzer) Update(rb *AfirRenderBuffer, delay int, hasDelay bool) {
	h := 0
	if hasDelay {
		h = 1
	}
	C.afir_rsa_update(a.p, rb.p, C.int(delay), C.int(h))
}
func (a *AfirRenderSignalAnalyzer) PoorSignalExcitation() bool {
	return C.afir_rsa_poor_signal_excitation(a.p) != 0
}
func (a *AfirRenderSignalAnalyzer) MaskRegionsAroundNarrowBands(v *[aec3.FftLengthBy2Plus1]float32) {
	C.afir_rsa_mask(a.p, afirF(&v[0]))
}
func (a *AfirRenderSignalAnalyzer) NarrowPeakBand() (int, bool) {
	var b C.int
	ok := C.afir_rsa_narrow_peak_band(a.p, &b) != 0
	return int(b), ok
}

// AfirCoarseGain wraps webrtc::CoarseFilterUpdateGain.
type AfirCoarseGain struct{ p *C.afir_coarse }

func NewAfirCoarseGain(c aec3.CoarseConfiguration, changeDuration int) *AfirCoarseGain {
	return &AfirCoarseGain{C.afir_coarse_new(C.int(c.LengthBlocks), C.float(c.Rate), C.float(c.NoiseGate), C.int(changeDuration))}
}
func (g *AfirCoarseGain) Free()                 { C.afir_coarse_free(g.p) }
func (g *AfirCoarseGain) HandleEchoPathChange() { C.afir_coarse_handle_echo_path_change(g.p) }
func (g *AfirCoarseGain) SetConfig(c aec3.CoarseConfiguration, immediate bool) {
	i := 0
	if immediate {
		i = 1
	}
	C.afir_coarse_set_config(g.p, C.int(c.LengthBlocks), C.float(c.Rate), C.float(c.NoiseGate), C.int(i))
}
func (g *AfirCoarseGain) Compute(X2 *[aec3.FftLengthBy2Plus1]float32, a *AfirRenderSignalAnalyzer, E *aec3.FftData, sizePartitions int, saturated bool, G *aec3.FftData) {
	s := 0
	if saturated {
		s = 1
	}
	C.afir_coarse_compute(g.p, afirF(&X2[0]), a.p, afirFftPtr(E), C.int(sizePartitions), C.int(s), afirFftPtr(G))
}

// AfirRefinedGain wraps webrtc::RefinedFilterUpdateGain.
type AfirRefinedGain struct{ p *C.afir_refined }

func afirRefinedCfg(c aec3.RefinedConfiguration) [6]float32 {
	return [6]float32{float32(c.LengthBlocks), c.LeakageConverged, c.LeakageDiverged, c.ErrorFloor, c.ErrorCeil, c.NoiseGate}
}

func NewAfirRefinedGain(c aec3.RefinedConfiguration, changeDuration int) *AfirRefinedGain {
	cfg := afirRefinedCfg(c)
	return &AfirRefinedGain{C.afir_refined_new(afirF(&cfg[0]), C.int(changeDuration))}
}
func (g *AfirRefinedGain) Free() { C.afir_refined_free(g.p) }
func (g *AfirRefinedGain) HandleEchoPathChange(v aec3.EchoPathVariability) {
	b := func(x bool) C.int {
		if x {
			return 1
		}
		return 0
	}
	C.afir_refined_handle_echo_path_change(g.p, b(v.GainChange), C.int(v.DelayChange), b(v.ClockDrift))
}
func (g *AfirRefinedGain) SetConfig(c aec3.RefinedConfiguration, immediate bool) {
	cfg := afirRefinedCfg(c)
	i := 0
	if immediate {
		i = 1
	}
	C.afir_refined_set_config(g.p, afirF(&cfg[0]), C.int(i))
}
func (g *AfirRefinedGain) Compute(X2 *[aec3.FftLengthBy2Plus1]float32, a *AfirRenderSignalAnalyzer, so *aec3.SubtractorOutput, erl []float32, sizePartitions int, saturated, disallowLeakageDiverged bool, G *aec3.FftData) {
	s, d := 0, 0
	if saturated {
		s = 1
	}
	if disallowLeakageDiverged {
		d = 1
	}
	C.afir_refined_compute(g.p, afirF(&X2[0]), a.p, afirFftPtr(&so.ERefinedFft), afirF(&so.E2Refined[0]), afirF(&so.E2Coarse[0]), afirF(&erl[0]), C.int(sizePartitions), C.int(s), C.int(d), afirFftPtr(G))
}
