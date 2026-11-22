package aec3

import (
	"reflect"
	"testing"
)

// TestSuppressorStatesPointerFree checks that the states of the suppressor
// components hold nothing the garbage collector reads.
func TestSuppressorStatesPointerFree(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeOf(suppressionGainState{}),
		reflect.TypeOf(dominantNearendState{}),
		reflect.TypeOf(cngState{}),
		reflect.TypeOf(suppressionFilterState{}),
	} {
		if p := pointerPath(typ, typ.Name()); p != "" {
			t.Errorf("%s holds a pointer: %s", typ.Name(), p)
		}
	}
}

// TestSuppressorViewsOfCore checks that the views a core binds reach what
// init set up: the state of more than inlineChannels channels and the
// optional subband nearend detector.
func TestSuppressorViewsOfCore(t *testing.T) {
	for _, subband := range []bool{false, true} {
		for _, numCh := range []int{1, inlineChannelLimit + 1} {
			cfg := DefaultEchoCanceller3Config()
			cfg.Suppressor.UseSubbandNearendDetection = subband
			standalone := NewEchoRemover(&cfg, 16000, 1, numCh)
			ec := NewEchoCanceller3(&cfg, nil, 16000, 1, numCh)
			er := &ec.blockProcessor.ownEchoRemover
			if got, want := er.suppressionGain.subbandNearendDetector != nil, subband; got != want {
				t.Errorf("subband %v, %d channels: subband nearend detector bound: %v", subband, numCh, got)
			}
			for _, c := range []struct {
				name      string
				got, want bool
			}{
				{"suppressionGain.more", er.suppressionGain.more != nil, standalone.suppressionGain.more != nil},
				{"dominantNearendDetector.more", er.suppressionGain.dominantNearendDetector.more != nil,
					standalone.suppressionGain.dominantNearendDetector.more != nil},
				{"cng.more", er.cng.more != nil, standalone.cng.more != nil},
				{"suppressionFilter.more", er.suppressionFilter.more != nil, standalone.suppressionFilter.more != nil},
			} {
				if c.got != c.want {
					t.Errorf("subband %v, %d channels: %s bound: %v, want %v", subband, numCh, c.name, c.got, c.want)
				}
			}
			if got, want := len(er.cng.NoiseSpectrum()), numCh; got != want {
				t.Errorf("subband %v, %d channels: %d noise spectra", subband, numCh, got)
			}
		}
	}
}
