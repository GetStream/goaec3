//go:build !goexperiment.simd || purego || !(arm64 || amd64)

package aec3

// dlyHasDecimateKernel reports whether dlyDecimateDs4 is an archsimd kernel.
const dlyHasDecimateKernel = false

func dlyDecimateDs4(*[BlockSize]float32, *[BlockSize / 4]float32, *[4]BiQuad) bool {
	panic("aec3: no decimation kernel")
}
