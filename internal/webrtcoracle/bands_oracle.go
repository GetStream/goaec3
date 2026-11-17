//go:build webrtc_oracle

package webrtcoracle

/*
#include "bands_oracle.h"
*/
import "C"

import "unsafe"

func bandsF(s []float32) *C.float { return (*C.float)(unsafe.Pointer(&s[0])) }
func bandsI(s []int16) *C.int16_t { return (*C.int16_t)(unsafe.Pointer(&s[0])) }

// BandsThreeBandFilterBank wraps webrtc::ThreeBandFilterBank.
type BandsThreeBandFilterBank struct{ p unsafe.Pointer }

func NewBandsThreeBandFilterBank() *BandsThreeBandFilterBank {
	return &BandsThreeBandFilterBank{p: C.bands_tbfb_new()}
}

// Analysis: in has 480 samples, out is [3][160] flattened.
func (f *BandsThreeBandFilterBank) Analysis(in, out []float32) {
	C.bands_tbfb_analysis(f.p, bandsF(in), bandsF(out))
}

// Synthesis: in is [3][160] flattened, out has 480 samples.
func (f *BandsThreeBandFilterBank) Synthesis(in, out []float32) {
	C.bands_tbfb_synthesis(f.p, bandsF(in), bandsF(out))
}

func (f *BandsThreeBandFilterBank) Free() { C.bands_tbfb_free(f.p) }

// BandsAnalysisQMF runs WebRtcSpl_AnalysisQMF.
func BandsAnalysisQMF(in, low, high []int16, s1, s2 *[6]int32) {
	C.bands_qmf_analysis(bandsI(in), C.size_t(len(in)), bandsI(low), bandsI(high),
		(*C.int32_t)(unsafe.Pointer(&s1[0])), (*C.int32_t)(unsafe.Pointer(&s2[0])))
}

// BandsSynthesisQMF runs WebRtcSpl_SynthesisQMF.
func BandsSynthesisQMF(low, high, out []int16, s1, s2 *[6]int32) {
	C.bands_qmf_synthesis(bandsI(low), bandsI(high), C.size_t(len(low)), bandsI(out),
		(*C.int32_t)(unsafe.Pointer(&s1[0])), (*C.int32_t)(unsafe.Pointer(&s2[0])))
}

func BandsFloatS16ToS16(src []float32, dst []int16) {
	C.bands_float_s16_to_s16(bandsF(src), C.size_t(len(src)), bandsI(dst))
}

func BandsS16ToFloatS16(src []int16, dst []float32) {
	C.bands_s16_to_float_s16(bandsI(src), C.size_t(len(src)), bandsF(dst))
}

func BandsFloatToFloatS16(src, dst []float32) {
	C.bands_float_to_float_s16(bandsF(src), C.size_t(len(src)), bandsF(dst))
}

func BandsFloatS16ToFloat(src, dst []float32) {
	C.bands_float_s16_to_float(bandsF(src), C.size_t(len(src)), bandsF(dst))
}

// BandsAudioBuffer wraps webrtc::AudioBuffer with equal input, buffer and
// output rates.
type BandsAudioBuffer struct{ p unsafe.Pointer }

func NewBandsAudioBuffer(rate, numCh int) *BandsAudioBuffer {
	return &BandsAudioBuffer{p: C.bands_ab_new(C.int(rate), C.int(numCh))}
}

// CopyFrom copies float [-1, 1] data ([ch][n] flattened) into the buffer.
func (a *BandsAudioBuffer) CopyFrom(data []float32) { C.bands_ab_copy_from(a.p, bandsF(data)) }
func (a *BandsAudioBuffer) Split()                  { C.bands_ab_split(a.p) }
func (a *BandsAudioBuffer) Merge()                  { C.bands_ab_merge(a.p) }

// ReadBands reads split bands as [band][ch][160] flattened.
func (a *BandsAudioBuffer) ReadBands(bands []float32)  { C.bands_ab_read_bands(a.p, bandsF(bands)) }
func (a *BandsAudioBuffer) WriteBands(bands []float32) { C.bands_ab_write_bands(a.p, bandsF(bands)) }

// ReadData reads the full-band FloatS16 data as [ch][n] flattened.
func (a *BandsAudioBuffer) ReadData(data []float32) { C.bands_ab_read_data(a.p, bandsF(data)) }
func (a *BandsAudioBuffer) Free()                   { C.bands_ab_free(a.p) }
