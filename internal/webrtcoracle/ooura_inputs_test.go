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

// oouraCheck requires the Go transforms of the build's lane to be
// bit-identical to ref on every input: ooura.Fft and InverseFft, which are
// the transcription of the lane, and the split transforms aec3 runs, single
// and in pairs, against ref with the FftData copy.
func oouraCheck(t *testing.T, ref func(a *[128]float32, inverse bool)) {
	t.Helper()
	same := func(what string, trial int, got, want []float32) {
		t.Helper()
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s input %d idx %d: got %v (%#x) want %v (%#x)", what, trial, i,
					got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
	// unpack is FftData::CopyFromPackedArray.
	unpack := func(v *[128]float32, re, im *[65]float32) {
		*re, *im = [65]float32{}, [65]float32{}
		re[0], re[64] = v[0], v[1]
		for k := 1; k < 64; k++ {
			re[k], im[k] = v[2*k], v[2*k+1]
		}
	}
	var s ooura.Scratch
	var ps ooura.PairScratch
	in := oouraInputs()
	for trial, a := range in {
		// The second frame of a pair.
		b := in[(trial*7+3)%len(in)]

		want, got := a, a
		ref(&want, false)
		ooura.Fft(&got)
		same("Fft", trial, got[:], want[:])
		var wre, wim, re, im, re1, im1 [65]float32
		unpack(&want, &wre, &wim)
		x := a
		ooura.FftSplit(&x, &re, &im, &s)
		same("FftSplit re", trial, re[:], wre[:])
		same("FftSplit im", trial, im[:], wim[:])
		x, y := b, a
		ooura.FftSplitPair(&x, &y, &re1, &im1, &re, &im, &ps)
		same("FftSplitPair re", trial, re[:], wre[:])
		same("FftSplitPair im", trial, im[:], wim[:])

		// The inverse of the input taken as bins: CopyToPackedArray of the
		// unpacked frame is the frame.
		want, got = a, a
		ref(&want, true)
		ooura.InverseFft(&got)
		same("InverseFft", trial, got[:], want[:])
		unpack(&a, &re, &im)
		unpack(&b, &re1, &im1)
		ooura.InverseFftSplit(&re, &im, &got, &s)
		same("InverseFftSplit", trial, got[:], want[:])
		ooura.InverseFftSplitPair(&re1, &im1, &re, &im, &x, &got, &ps)
		same("InverseFftSplitPair", trial, got[:], want[:])
	}
}
