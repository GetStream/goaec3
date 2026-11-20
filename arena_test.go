package goaec3

import (
	"reflect"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

// TestArenaClasses checks the table of arena classes: every class holds what
// its entry says, has no pointers, and is no smaller than the one before it,
// so that a canceller gets the smallest class that fits.
func TestArenaClasses(t *testing.T) {
	var last uintptr
	for i, c := range ecArenaClasses {
		a := c.alloc()
		v := a.Views()
		if len(v.F32) != c.f || len(v.Fft) != c.h || len(v.Spec) != c.s || len(v.I32) != c.i || len(a.splitters()) != c.p {
			t.Errorf("class %d: holds %d, %d, %d, %d, %d, the table says %d, %d, %d, %d, %d", i,
				len(v.F32), len(v.Fft), len(v.Spec), len(v.I32), len(a.splitters()), c.f, c.h, c.s, c.i, c.p)
		}
		typ := reflect.TypeOf(a).Elem()
		if words, _ := pointerWords(typ); words != 0 {
			t.Errorf("class %d: %d pointer words", i, words)
		}
		if typ.Size() < last {
			t.Errorf("class %d: %d bytes, the class before it has %d", i, typ.Size(), last)
		}
		last = typ.Size()
	}
}

// TestArenaClassOfDefaults checks that the default configs with up to two
// channels a side get one arena, of a class, also when they process stereo
// from the start.
func TestArenaClassOfDefaults(t *testing.T) {
	isClass := func(a aec3.Arena) bool {
		_, slices := a.(*ecSliceArena)
		return a != nil && !slices
	}
	for _, rate := range []int{16000, 32000, 48000} {
		for _, r := range []int{1, 2} {
			for _, c := range []int{1, 2} {
				for _, stereo := range []bool{false, true} {
					cfg, multi := DefaultConfig(), MultichannelConfig()
					// Without the detection, two render channels are
					// processed as stereo from the start.
					cfg.MultiChannel.DetectStereoContent = !stereo
					multi.MultiChannel.DetectStereoContent = !stereo
					e, err := NewWithMultichannelConfig(cfg, multi, rate, r, c)
					if err != nil {
						t.Fatal(err)
					}
					arena, bpArena := e.core.ArenasForTesting()
					if !isClass(arena) || bpArena != nil {
						t.Errorf("%d Hz, %d render, %d capture channels, stereo %v: arenas %T and %T",
							rate, r, c, stereo, arena, bpArena)
					}
				}
			}
		}
	}
}
