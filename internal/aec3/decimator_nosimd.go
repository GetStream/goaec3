//go:build !goexperiment.simd || purego || !arm64

package aec3

// dlyHasDecimateKernel is false: this build runs the scalar sections.
const dlyHasDecimateKernel = false

func dlyDecimateDs4(*[BlockSize]float32, *[BlockSize / 4]float32, *[4]BiQuad) bool {
	panic("aec3: no decimation kernel")
}
