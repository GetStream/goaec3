package aec3

import (
	"reflect"
	"testing"
)

// statePointers lists the fields of t that hold a pointer of any kind, by
// path. Fields whose type is in skip are not looked into.
func statePointers(t reflect.Type, path string, skip map[reflect.Type]bool, found *[]string) {
	if skip[t] {
		return
	}
	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Interface, reflect.Map, reflect.Chan, reflect.Func,
		reflect.String, reflect.UnsafePointer:
		*found = append(*found, path+" ("+t.String()+")")
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			statePointers(f.Type, path+"."+f.Name, skip, found)
		}
	case reflect.Array:
		if t.Len() > 0 {
			statePointers(t.Elem(), path+"[]", skip, found)
		}
	}
}

// TestRenderStatesPointerFree checks that the states of the render delay
// buffer, its rings and the alignment mixer hold no pointer, and those of
// the delay controller and its estimator none outside the matched filter
// and the lag aggregator.
func TestRenderStatesPointerFree(t *testing.T) {
	for _, tc := range []struct {
		state any
		skip  []any
	}{
		{state: renderDelayBufferState{}},
		{state: blockBufferState{}},
		{state: spectrumBufferState{}},
		{state: fftBufferState{}},
		{state: downsampledRenderBufferState{}},
		{state: alignmentMixerState{}},
		{state: renderDelayControllerState{}, skip: []any{MatchedFilter{}, MatchedFilterLagAggregator{}}},
		{state: echoPathDelayEstimatorState{}, skip: []any{MatchedFilter{}, MatchedFilterLagAggregator{}}},
	} {
		skip := map[reflect.Type]bool{}
		for _, s := range tc.skip {
			skip[reflect.TypeOf(s)] = true
		}
		typ := reflect.TypeOf(tc.state)
		var found []string
		statePointers(typ, typ.Name(), skip, &found)
		for _, f := range found {
			t.Errorf("%s holds a pointer", f)
		}
	}
}

// TestRenderDelayBufferRingsDisjoint checks that the slices a
// RenderDelayBuffer registers with a binder do not overlap: the binder moves
// each to its own part of an arena.
func TestRenderDelayBufferRingsDisjoint(t *testing.T) {
	cfg := DefaultEchoCanceller3Config()
	r := NewRenderDelayBuffer(&cfg, 48000, 2)
	blocks, low := r.blocks.data, r.lowRate.Buffer
	if len(blocks) == 0 || len(low) == 0 {
		t.Fatal("empty ring")
	}
	if cap(blocks) != len(blocks) {
		t.Errorf("the block ring has capacity %d beyond its %d samples", cap(blocks), len(blocks))
	}
	for i := range low {
		low[i] = 1
	}
	for i, v := range blocks {
		if v != 0 {
			t.Fatalf("sample %d of the block ring is part of the low-rate ring", i)
		}
	}
}
