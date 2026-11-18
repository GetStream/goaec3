package aec3

import (
	"math/rand"
	"testing"
)

// rbSpectralSumRef is the scalar SpectralSum loop: per bin, +0 plus every
// spectrum (all channels) from the read position in ring order.
func rbSpectralSumRef(sb *SpectrumBuffer, numSpectra int, X2 *[FftLengthBy2Plus1]float32) {
	*X2 = [FftLengthBy2Plus1]float32{}
	position := sb.Read
	for j := 0; j < numSpectra; j++ {
		for c := range sb.Buffer[position] {
			for k := range X2 {
				X2[k] += sb.Buffer[position][c][k]
			}
		}
		position = sb.IncIndex(position)
	}
}

func TestRenderBufferSpectralSums(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for trial := 0; trial < 500; trial++ {
		size := 10 + rng.Intn(40)
		nch := 1 + rng.Intn(4)
		sb := NewSpectrumBuffer(size, nch)
		for i := range sb.Buffer {
			for c := range sb.Buffer[i] {
				copy(sb.Buffer[i][c][:], kernEdgeVec(rng, FftLengthBy2Plus1))
			}
		}
		sb.Read = rng.Intn(size)
		rb := NewRenderBuffer(NewBlockBuffer(size, 1, nch), sb, NewFftBuffer(size, nch))
		short := rng.Intn(size + 1)
		long := short + rng.Intn(size-short+1)
		var got, want, got2, want2 [FftLengthBy2Plus1]float32
		rb.SpectralSum(long, &got)
		rbSpectralSumRef(sb, long, &want)
		kernCheckExact(t, "SpectralSum", got[:], want[:])
		rb.SpectralSums(short, long, &got, &got2)
		rbSpectralSumRef(sb, short, &want)
		rbSpectralSumRef(sb, long, &want2)
		kernCheckExact(t, "SpectralSums shorter", got[:], want[:])
		kernCheckExact(t, "SpectralSums longer", got2[:], want2[:])
	}
}
