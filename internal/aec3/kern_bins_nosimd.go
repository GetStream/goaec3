//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

// kern_bins.go helpers without archsimd: the references.

func kernScale(x []float32, a float32, z []float32) { kernScaleGeneric(x, a, z) }

func kernAdd(x, y, z []float32) { kernAddGeneric(x, y, z) }

func kernSub(x, y, z []float32) { kernSubGeneric(x, y, z) }

func kernSumRows(dst *[FftLengthBy2Plus1]float32, rows []*[FftLengthBy2Plus1]float32) {
	kernSumRowsGeneric(dst, rows)
}

func kernUnpack(v *[FftLength]float32, re, im *[FftLengthBy2Plus1]float32) {
	kernUnpackGeneric(v, re, im)
}

func kernPack(re, im *[FftLengthBy2Plus1]float32, v *[FftLength]float32) { kernPackGeneric(re, im, v) }

// kernMinMaxFast reports whether kernMinMax has a fast path.
const kernMinMaxFast = false

// kernMinMax defers to the caller's loop.
func kernMinMax(x []float32) (mn, mx float32, ok bool) { return 0, 0, false }

func kernClamp(x []float32, lo, hi float32) { kernClampGeneric(x, lo, hi) }
