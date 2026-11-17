//go:build webrtc_oracle

package webrtcoracle

import (
	"math"
	"math/rand"
	"testing"

	"github.com/thesyncim/goaec3/internal/ooura"
)

func TestOouraFftBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 2000; trial++ {
		var a [128]float32
		for i := range a {
			a[i] = float32(rng.NormFloat64() * 3000)
		}
		for _, inv := range []bool{false, true} {
			want, got := a, a
			OouraFft(&want, inv)
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
