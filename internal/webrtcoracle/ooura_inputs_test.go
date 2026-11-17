//go:build webrtc_oracle || (ooura_sse2_oracle && amd64)

package webrtcoracle

import (
	"math"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/ooura"
)

// oouraInputs returns random frames plus structured frames that exercise
// signed zeros, zero-padded halves (the ZeroPaddedFft layout), impulses,
// constants and subnormals.
func oouraInputs() [][128]float32 {
	rng := rand.New(rand.NewSource(1))
	negZero := math.Float32frombits(0x80000000)
	var out [][128]float32
	for trial := 0; trial < 2000; trial++ {
		var a [128]float32
		for i := range a {
			a[i] = float32(rng.NormFloat64() * 3000)
		}
		out = append(out, a)
	}
	// All zeros, all negative zeros, mixed signed zeros.
	var z, nz, mz [128]float32
	for i := range nz {
		nz[i] = negZero
		if rng.Intn(2) == 0 {
			mz[i] = negZero
		}
	}
	out = append(out, z, nz, mz)
	for trial := 0; trial < 200; trial++ {
		var lo, hi, sparse, quant [128]float32
		for i := 0; i < 64; i++ {
			hi[64+i] = float32(rng.NormFloat64() * 1000)
			lo[i] = float32(rng.NormFloat64() * 1000)
		}
		for i := range sparse {
			switch rng.Intn(4) {
			case 0:
				sparse[i] = float32(rng.NormFloat64())
			case 1:
				sparse[i] = negZero
			}
			// Small integers make many exact cancellations to signed zero.
			quant[i] = float32(rng.Intn(5) - 2)
		}
		out = append(out, lo, hi, sparse, quant)
	}
	for i := 0; i < 128; i++ {
		var imp, cst, sub [128]float32
		imp[i] = 1
		for k := range cst {
			cst[k] = float32(i - 64)
			sub[k] = math.Float32frombits(uint32(rng.Intn(1 << 23)))
			if rng.Intn(2) == 0 {
				sub[k] = -sub[k]
			}
		}
		out = append(out, imp, cst, sub)
	}
	return out
}

// oouraCheck runs the Go FFT (the build's lane) and ref on every input and
// requires bit-identical outputs.
func oouraCheck(t *testing.T, ref func(a *[128]float32, inverse bool)) {
	t.Helper()
	for trial, a := range oouraInputs() {
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
					t.Fatalf("input %d inv=%v idx %d: got %v (%#x) want %v (%#x)", trial, inv, i,
						got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
				}
			}
		}
	}
}
