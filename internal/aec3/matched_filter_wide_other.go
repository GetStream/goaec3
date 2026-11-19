//go:build !amd64 || goexperiment.simd || purego

package aec3

// dlyWide is empty in the builds that do not run the pure-Go references of
// the AVX2 lane in production (see matched_filter_wide_amd64.go).
type dlyWide struct{}

func (w *dlyWide) fill(x []float32) []float64 { return nil }
