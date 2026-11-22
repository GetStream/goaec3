//go:build !amd64

package bands

// bsNaNS16Vector is the int16 a NaN sample becomes in the vectorized
// FloatS16ToS16 loop on arm64: fcvtzs gives 0.
const bsNaNS16Vector int16 = 0

// bsIsAMD64 selects the pure-Go kernels written for amd64.
const bsIsAMD64 = false
