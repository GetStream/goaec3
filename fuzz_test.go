package goaec3

import (
	"testing"
)

// robustSeeds are hand-written fuzz inputs: header (rate, render channels,
// capture channels, flags, seed, delay, mutation count) followed by ops.
var robustSeeds = [][]byte{
	// 16 kHz mono, default config, 400 render/capture pairs.
	{0, 0, 0, 0, 1, 40, 0, robustOpPairs, 0, 199},
	// 48 kHz stereo render, mono capture, linear output, multichannel config.
	{2, 1, 0, 2 | 8, 2, 60, 0, 0, robustOpPairs, 1, 150, robustOpRenderBurst, 0, 120,
		robustOpCaptureBurst, 2 * robustSigFullScale, 30, robustOpPairs, 1, 100},
	// 32 kHz external delay with buffer delays and bursts.
	{1, 0, 0, 4, 3, 80, 0, robustOpSetDelay, 0, 2, 80, 0, robustOpPairs, 0, 60, robustOpRenderBurst, 0, 40,
		robustOpSetDelay, 0, 1, robustOpCaptureBurst, 0, 50, robustOpPairs, 0, 40},
	// Special values on both sides.
	{0, 0, 0, 0, 4, 20, 0, robustOpPairs, 0, 100,
		robustOpRender, 2 * robustSigInf, robustOpCapture, 2 * robustSigClipping,
		robustOpRender, 2 * robustSigDenormal, robustOpCapture, 2 * robustSigSatEdge,
		robustOpRender, 2 * robustSigNegZero, robustOpCapture, 2 * robustSigDC,
		robustOpGainChange, 0, robustOpOutputUsage, 0, robustOpPairs, 0, 20,
		robustOpRender, 2 * robustSigNaN, robustOpCapture, 2 * robustSigNaN, robustOpPairs, 0, 20},
	// Eight channels each way.
	{2, 207, 207, 0, 5, 10, 0, robustOpPairs, 0, 60},
}

// FuzzEchoCanceller drives the public API with fuzzed configs (any field
// over its full range, then Validate), rates, channel counts, call
// sequences and audio, including silence, full scale, clipping, denormals,
// ±Inf and NaN. It requires that nothing panics and that the output and the
// metrics are finite until a NaN sample is fed (FloatToFloatS16 clamps ±Inf,
// NaN propagates into the filter state, as in the C++).
func FuzzEchoCanceller(f *testing.F) {
	for _, s := range robustSeeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		p := robustDecode(data)
		robustRun(t, &p)
	})
}
