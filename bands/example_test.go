package bands_test

import (
	"fmt"

	"github.com/GetStream/goaec3/bands"
)

// A 10 ms frame at 48 kHz is split into three bands of 160 samples, which
// the submodules process, and merged back.
func ExampleBandSplitter() {
	const rate, channels = 48000, 1
	s := bands.NewBandSplitter(rate, channels)

	frame := [][]float32{make([]float32, rate/100)}
	split := make([][][]float32, s.NumBands())
	for b := range split {
		split[b] = [][]float32{make([]float32, s.NumFramesPerBand())}
	}

	// The filters work on FloatS16 samples.
	bands.FloatToFloatS16Slice(frame[0], frame[0])
	s.SplitIntoFrequencyBands(frame, split)
	// Process split[band][channel] here.
	s.MergeFrequencyBands(split, frame)
	bands.FloatS16ToFloatSlice(frame[0], frame[0])

	fmt.Println(len(split), len(split[0][0]), len(frame[0]))
	// Output: 3 160 480
}
