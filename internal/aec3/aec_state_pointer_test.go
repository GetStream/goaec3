package aec3

import (
	"reflect"
	"testing"
)

// pointerPath returns the path of the first field of t that the garbage
// collector reads (a pointer, slice, map, interface, string, func or chan),
// and "" when t has none.
func pointerPath(t reflect.Type, path string) string {
	switch t.Kind() {
	case reflect.Pointer, reflect.UnsafePointer, reflect.Slice, reflect.Map, reflect.Interface,
		reflect.String, reflect.Func, reflect.Chan:
		return path + " (" + t.String() + ")"
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if p := pointerPath(f.Type, path+"."+f.Name); p != "" {
				return p
			}
		}
	case reflect.Array:
		return pointerPath(t.Elem(), path+"[]")
	}
	return ""
}

// TestAecStatePointerFree checks that the state of an AecState and of its
// estimators and analyzers holds nothing the garbage collector reads.
func TestAecStatePointerFree(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeOf(aecStateState{}),
		reflect.TypeOf(subtractorOutputAnalyzerState{}),
		reflect.TypeOf(filterAnalyzerState{}),
		reflect.TypeOf(reverbModelEstimatorState{}),
		reflect.TypeOf(erleEstimatorState{}),
		reflect.TypeOf(fullBandErleEstimatorState{}),
		reflect.TypeOf(subbandErleEstimatorState{}),
	} {
		if p := pointerPath(typ, typ.Name()); p != "" {
			t.Errorf("%s holds a pointer: %s", typ.Name(), p)
		}
	}
}

// TestAecStateBoundView checks that an AecState bound to the arenas of a
// canceller behaves as one that stands alone: the views of a core are bound
// anew and compared with the state of the AecState that init set up.
func TestAecStateBoundView(t *testing.T) {
	for _, numCh := range []int{1, 2, 3} {
		for _, sections := range []int{1, 2} {
			config := DefaultEchoCanceller3Config()
			config.Erle.NumSections = sections
			var c Ec3Core
			c.Init(&config, nil, 16000, numCh, numCh, 0)
			var e EchoCanceller3
			c.Bind(&e)
			a := &e.blockProcessor.ownEchoRemover.aecState
			alone := NewAecState(&config, numCh)
			if got, want := len(a.filterAnalyzer.storage), len(alone.filterAnalyzer.storage); got != want {
				t.Errorf("%d channels: filter analyzer storage of %d samples, want %d", numCh, got, want)
			}
			if got, want := len(a.reverbModelEstimator.storage), len(alone.reverbModelEstimator.storage); got != want {
				t.Errorf("%d channels: reverb model storage of %d samples, want %d", numCh, got, want)
			}
			if n := len(a.filterAnalyzer.storage); n > 0 && len(a.reverbModelEstimator.storage) > 0 {
				// The two storages are disjoint: a write to the end of one
				// does not show in the other.
				a.filterAnalyzer.storage[n-1] = 1
				if a.reverbModelEstimator.storage[0] != 0 {
					t.Errorf("%d channels: the storages of the analyzers overlap", numCh)
				}
				a.filterAnalyzer.storage[n-1] = 0
			}
			if (a.more != nil) != (alone.more != nil) ||
				(a.subtractorOutputAnalyzer.more != nil) != (alone.subtractorOutputAnalyzer.more != nil) ||
				(a.filterAnalyzer.more != nil) != (alone.filterAnalyzer.more != nil) ||
				(a.reverbModelEstimator.more != nil) != (alone.reverbModelEstimator.more != nil) ||
				(a.erleEstimator.fullbandErleEstimator.more != nil) != (alone.erleEstimator.fullbandErleEstimator.more != nil) ||
				(a.erleEstimator.subbandErleEstimator.more != nil) != (alone.erleEstimator.subbandErleEstimator.more != nil) ||
				(a.erleEstimator.signalDependentErleEstimator != nil) != (sections > 1) {
				t.Errorf("%d channels, %d sections: the pointers of the bound view differ from those of init", numCh, sections)
			}
			if !reflect.DeepEqual(*a.aecStateState, *alone.aecStateState) {
				// The spans differ by construction; compare without them.
				x, y := *a.aecStateState, *alone.aecStateState
				x.filterAnalyzer.storageSpan, y.filterAnalyzer.storageSpan = span{}, span{}
				x.reverbModelEstimator.storageSpan, y.reverbModelEstimator.storageSpan = span{}, span{}
				if !reflect.DeepEqual(x, y) {
					t.Errorf("%d channels, %d sections: the state of the bound view differs from that of init", numCh, sections)
				}
			}
		}
	}
}
