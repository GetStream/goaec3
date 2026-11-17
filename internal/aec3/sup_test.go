package aec3

import "testing"

func TestSupAllocs(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	const numCh = 2
	Y2 := make([][FftLengthBy2Plus1]float32, numCh)
	for ch := range Y2 {
		for k := range Y2[ch] {
			Y2[ch][k] = float32(1000 + k*ch)
		}
	}
	low, high := make([]FftData, numCh), make([]FftData, numCh)
	cng := NewComfortNoiseGenerator(&cfg, DetectOptimization(), numCh)
	sf := NewSuppressionFilter(DetectOptimization(), 48000, numCh)
	var gain [FftLengthBy2Plus1]float32
	for k := range gain {
		gain[k] = 0.5
	}
	e := NewBlock(3, numCh)
	dn := NewDominantNearendDetector(&cfg.Suppressor.DominantNearendDetection, numCh)
	sn := cfg.Suppressor.SubbandNearendDetection
	sn.NearendAverageBlocks = 3
	sn.Subband1 = SubbandRegion{Low: 1, High: 10}
	sn.Subband2 = SubbandRegion{Low: 20, High: 40}
	snd := NewSubbandNearendDetector(&sn, numCh)
	mccd := NewMultiChannelContentDetector(true, 2, 0, 300, 2)
	frame := [][][]float32{{make([]float32, 160), make([]float32, 160)}}
	var dets []NearendDetector = []NearendDetector{dn, snd}

	if a := testing.AllocsPerRun(200, func() {
		cng.Compute(false, Y2, low, high)
		sf.ApplyGain(low, high, &gain, 0.3, low, &e)
		for _, d := range dets {
			d.Update(Y2, Y2, cng.NoiseSpectrum(), false)
		}
		mccd.UpdateDetection(frame)
	}); a != 0 {
		t.Fatalf("%v allocs per block", a)
	}
}
