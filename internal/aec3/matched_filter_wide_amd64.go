//go:build amd64 && !goexperiment.simd && !purego

package aec3

// dlyWideCap is the size of the downsampled render buffer of the default
// delay configuration, GetDownSampledBufferSize(4, 5). Larger buffers are
// not widened.
const dlyWideCap = 2448

// dlyWide holds the downsampled render buffer widened to float64 for the
// pure-Go references of the AVX2 lane, which this build runs on CPUs with
// AVX2. A render sample enters many software fused multiply-adds per block,
// which work in float64. dlyWide has no pointers.
type dlyWide struct {
	x [dlyWideCap]float64
}

// fill widens x, the whole render buffer, and returns the widened samples,
// or nil if x is too large.
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
