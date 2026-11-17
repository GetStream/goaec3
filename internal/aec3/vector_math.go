package aec3

// VectorMath mirrors aec3::VectorMath. Every lane (scalar, SSE2, AVX2, NEON
// on arm64) computes the same correctly rounded per-element results, so the
// optimization value only selects the kernel.
type VectorMath struct {
	optimization Aec3Optimization
}

func NewVectorMath(opt Aec3Optimization) VectorMath { return VectorMath{optimization: opt} }

// Sqrt replaces each element with its square root.
func (v VectorMath) Sqrt(x []float32) { kernVectorSqrt(x) }

// Multiply sets z = x*y element-wise.
func (v VectorMath) Multiply(x, y, z []float32) {
	vectorMultiply(x, y, z)
}

// Accumulate sets z += x element-wise.
func (v VectorMath) Accumulate(x, z []float32) {
	vectorAccumulate(x, z)
}

func vectorMultiplyGeneric(x, y, z []float32) {
	y = y[:len(x)]
	z = z[:len(x)]
	for i := range x {
		z[i] = x[i] * y[i]
	}
}

func vectorAccumulateGeneric(x, z []float32) {
	z = z[:len(x)]
	for i := range x {
		z[i] += x[i]
	}
}
