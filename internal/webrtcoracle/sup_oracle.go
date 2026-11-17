//go:build webrtc_oracle

package webrtcoracle

/*
#include "sup_oracle.h"
*/
import "C"

import (
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

func supBool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func supSpec(s [][aec3.FftLengthBy2Plus1]float32) *C.float {
	return (*C.float)(unsafe.Pointer(&s[0][0]))
}

func supFft(s []aec3.FftData) *C.float { return (*C.float)(unsafe.Pointer(&s[0].Re[0])) }

// SupComfortNoiseGenerator wraps webrtc::ComfortNoiseGenerator.
type SupComfortNoiseGenerator struct {
	p     unsafe.Pointer
	numCh int
}

func NewSupComfortNoiseGenerator(noiseFloorDbfs float32, opt aec3.Aec3Optimization, numCh int) *SupComfortNoiseGenerator {
	return &SupComfortNoiseGenerator{p: C.sup_cng_new(C.float(noiseFloorDbfs), C.int(opt), C.int(numCh)), numCh: numCh}
}

func (c *SupComfortNoiseGenerator) Compute(saturated bool, Y2 [][aec3.FftLengthBy2Plus1]float32,
	low, high []aec3.FftData, N2 [][aec3.FftLengthBy2Plus1]float32) {
	C.sup_cng_compute(c.p, supBool(saturated), supSpec(Y2), supFft(low), supFft(high), supSpec(N2))
}

func (c *SupComfortNoiseGenerator) Free() { C.sup_cng_free(c.p) }

// SupSuppressionFilter wraps webrtc::SuppressionFilter (or the scalar
// from-source build when scalar is true).
type SupSuppressionFilter struct {
	p      unsafe.Pointer
	scalar bool
}

func NewSupSuppressionFilter(opt aec3.Aec3Optimization, rate, numCh int, scalar bool) *SupSuppressionFilter {
	return &SupSuppressionFilter{p: C.sup_sf_new(C.int(opt), C.int(rate), C.int(numCh), supBool(scalar)), scalar: scalar}
}

// ApplyGain runs ApplyGain; e is [band][ch][64] flattened, read and written.
func (f *SupSuppressionFilter) ApplyGain(cn, cnHigh []aec3.FftData, gain *[aec3.FftLengthBy2Plus1]float32,
	highBandsGain float32, E []aec3.FftData, e []float32, numBands, numCh int) {
	C.sup_sf_apply(f.p, supFft(cn), supFft(cnHigh), (*C.float)(unsafe.Pointer(&gain[0])), C.float(highBandsGain),
		supFft(E), (*C.float)(unsafe.Pointer(&e[0])), C.int(numBands), C.int(numCh), supBool(f.scalar))
}

func (f *SupSuppressionFilter) Free() { C.sup_sf_free(f.p, supBool(f.scalar)) }

// SupNearendDetector wraps a webrtc::NearendDetector.
type SupNearendDetector struct{ p unsafe.Pointer }

func NewSupDominantNearendDetector(c *aec3.DominantNearendDetectionConfig, numCh int) *SupNearendDetector {
	return &SupNearendDetector{p: C.sup_dne_new(C.float(c.EnrThreshold), C.float(c.EnrExitThreshold),
		C.float(c.SnrThreshold), C.int(c.HoldDuration), C.int(c.TriggerThreshold),
		supBool(c.UseDuringInitialPhase), C.int(numCh))}
}

func NewSupSubbandNearendDetector(c *aec3.SubbandNearendDetectionConfig, numCh int) *SupNearendDetector {
	return &SupNearendDetector{p: C.sup_sne_new(C.int(c.NearendAverageBlocks), C.int(c.Subband1.Low),
		C.int(c.Subband1.High), C.int(c.Subband2.Low), C.int(c.Subband2.High),
		C.float(c.NearendThreshold), C.float(c.SnrThreshold), C.int(numCh))}
}

// Update runs Update and returns IsNearendState.
func (d *SupNearendDetector) Update(nearend, echo, noise [][aec3.FftLengthBy2Plus1]float32, initialState bool) bool {
	return C.sup_ne_update(d.p, supSpec(nearend), supSpec(echo), supSpec(noise), C.int(len(nearend)),
		supBool(initialState)) != 0
}

func (d *SupNearendDetector) Free() { C.sup_ne_free(d.p) }

// SupMultiChannelContentDetector wraps webrtc::MultiChannelContentDetector.
type SupMultiChannelContentDetector struct{ p unsafe.Pointer }

func NewSupMultiChannelContentDetector(detect bool, numRenderCh int, threshold float32,
	timeoutSeconds int, hysteresisSeconds float32) *SupMultiChannelContentDetector {
	return &SupMultiChannelContentDetector{p: C.sup_mccd_new(supBool(detect), C.int(numRenderCh),
		C.float(threshold), C.int(timeoutSeconds), C.float(hysteresisSeconds))}
}

// UpdateDetection takes frame as [band][ch][n] flattened and returns
// (changed, proper, temporary).
func (d *SupMultiChannelContentDetector) UpdateDetection(frame []float32, numBands, numCh, n int) (bool, bool, bool) {
	var proper, temporary C.int
	changed := C.sup_mccd_update(d.p, (*C.float)(unsafe.Pointer(&frame[0])), C.int(numBands), C.int(numCh),
		C.int(n), &proper, &temporary)
	return changed != 0, proper != 0, temporary != 0
}

func (d *SupMultiChannelContentDetector) Free() { C.sup_mccd_free(d.p) }

// SupConfigSelector wraps webrtc::ConfigSelector. The multichannel config,
// when present, differs from the mono one in filter.refined.length_blocks.
type SupConfigSelector struct{ p unsafe.Pointer }

func NewSupConfigSelector(detectStereo, hasMultichannel bool, mcMarker, numRenderCh int) *SupConfigSelector {
	return &SupConfigSelector{p: C.sup_cs_new(supBool(detectStereo), supBool(hasMultichannel),
		C.int(mcMarker), C.int(numRenderCh))}
}

func (s *SupConfigSelector) Update(mc bool)    { C.sup_cs_update(s.p, supBool(mc)) }
func (s *SupConfigSelector) ActiveMarker() int { return int(C.sup_cs_active_marker(s.p)) }
func (s *SupConfigSelector) Free()             { C.sup_cs_free(s.p) }
