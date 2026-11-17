package aec3

// Kernel entry points. SIMD-backed builds replace these via the per-arch
// files; until a kernel has an assembly implementation it routes here.

func fftDataSpectrum(_ Aec3Optimization, re, im, out *[FftLengthBy2Plus1]float32) {
	fftDataSpectrumGeneric(re, im, out)
}

func vectorMultiply(x, y, z []float32) { vectorMultiplyGeneric(x, y, z) }

func vectorAccumulate(x, z []float32) { vectorAccumulateGeneric(x, z) }
