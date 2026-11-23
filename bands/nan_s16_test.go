package bands

import (
	"math"
	"testing"
)

// TestFloatS16ToS16SliceNaN pins the NaN result of the slice conversion to
// the compiled C++ loop: bsNaNS16Vector in the eight-sample vector body, 0
// in the scalar remainder.
func TestFloatS16ToS16SliceNaN(t *testing.T) {
	nan := float32(math.NaN())
	for _, n := range []int{1, 7, 8, 9, 16, 17, 23, 160, 320} {
		src := make([]float32, n)
		for i := range src {
			src[i] = nan
			if i%2 == 1 {
				src[i] = -nan
			}
		}
		for _, f := range []func([]float32, []int16){FloatS16ToS16Slice, floatS16ToS16SliceRef} {
			dest := make([]int16, n)
			f(src, dest)
			for i, d := range dest {
				want := int16(0)
				if i < n&^7 {
					want = bsNaNS16Vector
				}
				if d != want {
					t.Fatalf("n %d idx %d: %d, want %d", n, i, d, want)
				}
			}
		}
	}
}
