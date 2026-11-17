//go:build webrtc_oracle

package webrtcoracle

import (
	"math"
	"math/rand"
	"testing"

	"github.com/thesyncim/goaec3/internal/ooura"
)

// TestOouraFftBitExact compares the Go FFT with the lane the build selects:
// the scalar C path under purego, the prebuilt NEON path otherwise.
func TestOouraFftBitExact(t *testing.T) {
	ref := OouraFft
	if purego {
		ref = OouraFftScalar
	}
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 2000; trial++ {
		var a [128]float32
		for i := range a {
			a[i] = float32(rng.NormFloat64() * 3000)
		}
		for _, inv := range []bool{false, true} {
			want, got := a, a
			ref(&want, inv)
			if inv {
				ooura.InverseFft(&got)
			} else {
				ooura.Fft(&got)
			}
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("trial %d inv=%v idx %d: got %v want %v", trial, inv, i, got[i], want[i])
				}
			}
		}
	}
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
