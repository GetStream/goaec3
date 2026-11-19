package goaec3

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
)

var footprintCases = []struct {
	name     string
	rate, ch int
}{
	{"16kHz_mono", 16000, 1},
	{"48kHz_mono", 48000, 1},
	{"48kHz_stereo", 48000, 2},
}

var footprintSink []*EchoCanceller

// footprint returns the heap bytes and objects that stay live per New.
func footprint(t testing.TB, rate, ch int) (bytes, objects uint64) {
	const n = 50
	footprintSink = make([]*EchoCanceller, 0, n)
	runtime.GC()
	var a, b runtime.MemStats
	runtime.ReadMemStats(&a)
	for i := 0; i < n; i++ {
		ec, err := New(DefaultConfig(), rate, ch, ch)
		if err != nil {
			t.Fatal(err)
		}
		footprintSink = append(footprintSink, ec)
	}
	runtime.GC()
	runtime.ReadMemStats(&b)
	runtime.KeepAlive(footprintSink)
	bytes = (b.HeapAlloc - a.HeapAlloc) / n
	objects = (b.HeapObjects - a.HeapObjects) / n
	footprintSink = nil
	return bytes, objects
}

// footprintObjectLimits bounds the heap objects per canceller of each
// footprintCases entry.
var footprintObjectLimits = []uint64{63, 69, 80}

// TestNewFootprint reports the live heap per canceller and bounds the number
// of heap objects, which sets most of the mark cost of a GC cycle with many
// live cancellers: marking an object costs several times as much as following
// another pointer into an object that is already marked.
func TestNewFootprint(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	for i, c := range footprintCases {
		bytes, objects := footprint(t, c.rate, c.ch)
		t.Logf("%s: %d B, %d objects per canceller", c.name, bytes, objects)
		if objects > footprintObjectLimits[i] {
			t.Errorf("%s: %d heap objects per canceller, want at most %d", c.name, objects, footprintObjectLimits[i])
		}
	}
}

// pointerSlots counts the non-nil pointers (pointers, slices, interfaces,
// maps, channels and functions) reachable from v: the words the garbage
// collector follows when it marks the value.
func pointerSlots(v reflect.Value, seen map[uintptr]bool) int {
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return 0
		}
		if seen[v.Pointer()] {
			return 1
		}
		seen[v.Pointer()] = true
		return 1 + pointerSlots(v.Elem(), seen)
	case reflect.Slice:
		if v.IsNil() || v.Cap() == 0 {
			return 0
		}
		n := 1
		if hasPointers(v.Type().Elem()) {
			for i := 0; i < v.Len(); i++ {
				n += pointerSlots(v.Index(i), seen)
			}
		}
		return n
	case reflect.Struct:
		n := 0
		for i := 0; i < v.NumField(); i++ {
			n += pointerSlots(v.Field(i), seen)
		}
		return n
	case reflect.Array:
		n := 0
		if hasPointers(v.Type().Elem()) {
			for i := 0; i < v.Len(); i++ {
				n += pointerSlots(v.Index(i), seen)
			}
		}
		return n
	case reflect.Interface:
		if v.IsNil() {
			return 0
		}
		return 1 + pointerSlots(v.Elem(), seen)
	case reflect.Map, reflect.Chan, reflect.Func:
		if v.IsNil() {
			return 0
		}
		return 1
	}
	return 0
}

func hasPointers(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Interface, reflect.Map, reflect.Chan, reflect.Func,
		reflect.String, reflect.UnsafePointer:
		return true
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if hasPointers(t.Field(i).Type) {
				return true
			}
		}
	case reflect.Array:
		return hasPointers(t.Elem())
	}
	return false
}

// TestPointerSlots bounds the pointers the garbage collector follows per
// canceller. The frames, the framing buffers, the render rings, the matched
// filters and the adaptive filter coefficients are pointer-free arrays with
// computed views, so the count does not grow with the buffer sizes. Many of
// the remaining pointers are slices of a component over arrays of its own
// struct (see channelSlice in internal/aec3), which lead to no other object.
func TestPointerSlots(t *testing.T) {
	limits := []int{124, 130, 150}
	for i, c := range footprintCases {
		ec, err := New(DefaultConfig(), c.rate, c.ch, c.ch)
		if err != nil {
			t.Fatal(err)
		}
		n := pointerSlots(reflect.ValueOf(ec), map[uintptr]bool{})
		t.Logf("%s: %d pointer slots per canceller", c.name, n)
		if n > limits[i] {
			t.Errorf("%s: %d pointer slots per canceller, want at most %d", c.name, n, limits[i])
		}
	}
}

// TestRenderBurstAllocation checks that render bursts allocate only when one
// is deeper than any before, and then once.
func TestRenderBurstAllocation(t *testing.T) {
	for _, c := range footprintCases {
		ec, err := New(DefaultConfig(), c.rate, c.ch, c.ch)
		if err != nil {
			t.Fatal(err)
		}
		n := c.rate / 100
		render := make([][]float32, c.ch)
		capture := make([][]float32, c.ch)
		for ch := range render {
			render[ch] = make([]float32, n)
			capture[ch] = make([]float32, n)
			for i := range render[ch] {
				render[ch][i] = float32((i*7919)%200-100) / 400
			}
		}
		burst := func(depth int) func() {
			return func() {
				for range depth {
					_ = ec.AnalyzeRender(render)
				}
				for range depth {
					_ = ec.ProcessCapture(capture)
				}
			}
		}
		for range 50 {
			burst(1)()
		}
		burst(12)()
		for _, depth := range []int{1, 5, 12} {
			if a := testing.AllocsPerRun(20, burst(depth)); a != 0 {
				t.Errorf("%s: render burst of %d after one of 12: %v allocations", c.name, depth, a)
			}
		}
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		burst(40)()
		runtime.ReadMemStats(&after)
		if after.Mallocs == before.Mallocs {
			t.Errorf("%s: a render burst deeper than any before did not grow the queue", c.name)
		}
		if a := testing.AllocsPerRun(20, burst(40)); a != 0 {
			t.Errorf("%s: second render burst of 40: %v allocations", c.name, a)
		}
	}
}

// BenchmarkNew reports the bytes and objects allocated per New, and the live
// heap per canceller.
func BenchmarkNew(b *testing.B) {
	for _, c := range footprintCases {
		b.Run(c.name, func(b *testing.B) {
			bytes, objects := footprint(b, c.rate, c.ch)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := New(DefaultConfig(), c.rate, c.ch, c.ch); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(bytes), "live-B")
			b.ReportMetric(float64(objects), "live-objects")
		})
	}
}

// BenchmarkGCMark measures one full GC cycle with 1000 live cancellers. With
// -cpu 1 the time is close to the mark work; with more Ps it is mostly the
// coordination of the mark workers.
func BenchmarkGCMark(b *testing.B) {
	for _, c := range footprintCases {
		b.Run(fmt.Sprintf("%s_x1000", c.name), func(b *testing.B) {
			live := make([]*EchoCanceller, 1000)
			for i := range live {
				ec, err := New(DefaultConfig(), c.rate, c.ch, c.ch)
				if err != nil {
					b.Fatal(err)
				}
				live[i] = ec
			}
			runtime.GC()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runtime.GC()
			}
			b.StopTimer()
			runtime.KeepAlive(live)
		})
	}
}
