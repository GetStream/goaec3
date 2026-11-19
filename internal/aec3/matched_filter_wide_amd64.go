//go:build amd64 && !goexperiment.simd && !purego

package aec3

// dlyWideCap is the size of the downsampled render buffer of the default
// delay configuration (GetDownSampledBufferSize(4, 5)); larger buffers are
// not widened.
const dlyWideCap = 2448

// dlyWide holds the downsampled render buffer widened to float64 for the
// pure-Go references of the AVX2 lane, which this build runs on CPUs with
// AVX2. Their software fused multiply-add works in float64, and every
// render sample enters about 160 of them per block (two per capture sample
// and filter): widening the buffer once per block takes two conversions and
// two loads per coefficient out of the filter loops. It has no pointers; the
// block processor keeps it behind the last pointer of the canceller.
type dlyWide struct {
	x [dlyWideCap]float64
}

// fill widens x, the whole render buffer, and returns the widened samples,
// or nil if the buffer is too large.
func (w *dlyWide) fill(x []float32) []float64 {
	if len(x) > dlyWideCap {
		return nil
	}
	xw := w.x[:len(x)]
	for i, v := range x {
		xw[i] = float64(v)
	}
	return xw
}
