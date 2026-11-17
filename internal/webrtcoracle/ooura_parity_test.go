//go:build webrtc_oracle

package webrtcoracle

import (
	"math"
	"math/rand"
	"testing"
)

// TestOouraFftBitExact compares the Go FFT with the lane the build selects:
// the scalar C path under purego, the prebuilt NEON path otherwise.
func TestOouraFftBitExact(t *testing.T) {
	ref := OouraFft
	if purego {
		ref = OouraFftScalar
	}
	oouraCheck(t, ref)
}

func TestExp2fInterposed(t *testing.T) {
	// Apple libm exp2f differs from the correctly rounded value on ~0.6% of
	// inputs, so 100k samples prove the interposer is active.
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 100000; i++ {
		x := float32(rng.NormFloat64() * 20)
		if got, want := cExp2f(x), float32(math.Exp2(float64(x))); got != want {
			t.Fatalf("exp2f(%v) = %v, want Go value %v", x, got, want)
		}
	}
}
