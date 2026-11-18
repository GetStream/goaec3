//go:build webrtc_src_oracle && !webrtc_oracle

package webrtcoracle

/*
#include "src_kern.h"
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/GetStream/goaec3/internal/aec3"
)

func srcF(p *float32) *C.float { return (*C.float)(unsafe.Pointer(p)) }

func srcFs(x []float32) *C.float {
	if len(x) == 0 {
		return nil
	}
	return srcF(&x[0])
}

func srcCheck(rc C.int, lane aec3.Aec3Optimization) {
	if rc != 0 {
		panic(fmt.Sprintf("webrtc_src_oracle: lane %d is not in this build", lane))
	}
}

// SrcKernHasLane reports whether the source build contains the kernels of
// lane (scalar everywhere, SSE2/AVX2 on amd64, NEON on arm64 unless purego).
func SrcKernHasLane(lane aec3.Aec3Optimization) bool { return C.src_kern_has_lane(C.int(lane)) != 0 }

// SrcMatchedFilterCore runs the lane's C++ MatchedFilterCore in place on h,
// errorSum, filtersUpdated and accumulatedError.
func SrcMatchedFilterCore(lane aec3.Aec3Optimization, xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool,
	accumulatedError, scratch []float32) {
	upd := ec3B(*filtersUpdated)
	es := C.float(*errorSum)
	srcCheck(C.src_kern_mf_core(C.int(lane), C.int(xStartIndex), C.float(x2SumThreshold), C.float(smoothing),
		srcFs(x), C.int(len(x)), srcFs(y), C.int(len(y)), srcFs(h), C.int(len(h)), &upd, &es,
		ec3B(computeAccumulatedError), srcFs(accumulatedError), C.int(len(accumulatedError)),
		srcFs(scratch), C.int(len(scratch))), lane)
	*filtersUpdated = upd != 0
	*errorSum = float32(es)
}

// SrcRenderBuffer is a C++ RenderBuffer whose FFT buffer the test fills.
type SrcRenderBuffer struct{ p *C.src_kern_rb }

func NewSrcRenderBuffer(size, numChannels int) *SrcRenderBuffer {
	return &SrcRenderBuffer{C.src_kern_rb_new(C.int(size), C.int(numChannels))}
}

func (r *SrcRenderBuffer) Free() { C.src_kern_rb_free(r.p) }

func (r *SrcRenderBuffer) SetFft(index, ch int, d *aec3.FftData) {
	C.src_kern_rb_set_fft(r.p, C.int(index), C.int(ch), srcF(&d.Re[0]))
}

func (r *SrcRenderBuffer) SetRead(read int) { C.src_kern_rb_set_read(r.p, C.int(read)) }

// srcFlatH packs H[partition][channel] contiguously.
func srcFlatH(H aec3.FirCoefficients) []aec3.FftData {
	return append([]aec3.FftData(nil), H.Flat()...)
}

func SrcApplyFilter(rb *SrcRenderBuffer, lane aec3.Aec3Optimization, numPartitions int, H []aec3.FftData,
	hPartitions, numChannels int, S *aec3.FftData) {
	srcCheck(C.src_kern_apply_filter(rb.p, C.int(lane), C.int(numPartitions), srcF(&H[0].Re[0]), C.int(hPartitions),
		C.int(numChannels), srcF(&S.Re[0])), lane)
}

func SrcAdaptPartitions(rb *SrcRenderBuffer, lane aec3.Aec3Optimization, G *aec3.FftData, numPartitions int,
	H []aec3.FftData, hPartitions, numChannels int) {
	srcCheck(C.src_kern_adapt_partitions(rb.p, C.int(lane), srcF(&G.Re[0]), C.int(numPartitions),
		srcF(&H[0].Re[0]), C.int(hPartitions), C.int(numChannels)), lane)
}

func SrcFreqResp(lane aec3.Aec3Optimization, numPartitions int, H []aec3.FftData, hPartitions, numChannels int,
	H2 [][aec3.FftLengthBy2Plus1]float32) {
	srcCheck(C.src_kern_freq_resp(C.int(lane), C.int(numPartitions), srcF(&H[0].Re[0]), C.int(hPartitions),
		C.int(numChannels), srcF(&H2[0][0])), lane)
}

func SrcErl(lane aec3.Aec3Optimization, H2 [][aec3.FftLengthBy2Plus1]float32, erl []float32) {
	var p *C.float
	if len(H2) > 0 {
		p = srcF(&H2[0][0])
	}
	srcCheck(C.src_kern_erl(C.int(lane), p, C.int(len(H2)), srcF(&erl[0])), lane)
}
