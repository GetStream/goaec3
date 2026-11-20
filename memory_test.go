package goaec3

import (
	"fmt"
	"math"
	"reflect"
	"runtime"
	"runtime/metrics"
	"strconv"
	"testing"
	"time"
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
var footprintObjectLimits = []uint64{2, 2, 3}

// TestNewFootprint reports the live heap per canceller and bounds the number
// of heap objects, which sets the mark cost of a GC cycle with many live
// cancellers: the canceller itself, of which the garbage collector reads the
// first few words, and its arena, which has no pointers; with two render
// channels the block processor has an arena of its own.
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

// TestPointerSlots bounds the pointers per canceller, as reflection counts
// them: the canceller, and an interface and its pointer per arena. The
// garbage collector follows one pointer per arena.
func TestPointerSlots(t *testing.T) {
	limits := []int{3, 3, 5}
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

// pointerWords returns the number of pointer words of a value of type t (a
// slice has one, an interface two) and the offset after the last of them.
func pointerWords(t reflect.Type) (words int, end uintptr) {
	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.String, reflect.Map, reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return 1, 8
	case reflect.Interface:
		return 2, 16
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if n, e := pointerWords(f.Type); n > 0 {
				words += n
				end = f.Offset + e
			}
		}
	case reflect.Array:
		if n, e := pointerWords(t.Elem()); n > 0 {
			words = n * t.Len()
			end = uintptr(t.Len()-1)*t.Elem().Size() + e
		}
	}
	return words, end
}

// TestPointerWords bounds what the garbage collector reads of a canceller,
// whatever its configuration: it reads every pointer word of the object, nil
// or not, and walks the object up to the last of them. The pointers of a
// canceller are its first words; the state of every component follows and
// holds none.
func TestPointerWords(t *testing.T) {
	if strconv.IntSize != 64 {
		t.Skip("the bounds are for 64-bit pointers")
	}
	const maxWords, maxEnd = 7, 136
	typ := reflect.TypeOf(EchoCanceller{})
	words, end := pointerWords(typ)
	t.Logf("%d pointer words, the last one ends at byte %d of %d", words, end, typ.Size())
	if words > maxWords {
		t.Errorf("%d pointer words per canceller, want at most %d", words, maxWords)
	}
	if end > maxEnd {
		t.Errorf("the pointers of a canceller end at byte %d of %d, want at most %d", end, typ.Size(), maxEnd)
	}
}

// TestCaptureChannelStates checks that the per-channel state of the capture
// channels behaves alike whether it is part of the components (up to two
// channels) or allocated (more): with the same signal in every capture
// channel, the linear output of the first channel is that of a canceller
// with one capture channel. (This holds for the first channel and the linear
// output only: the channels are not processed independently of each other,
// and after the suppressor they draw their comfort noise from one
// generator.)
func TestCaptureChannelStates(t *testing.T) {
	frames := 400
	if testing.Short() {
		frames = 150
	}
	for _, rate := range []int{16000, 48000} {
		n := rate / 100
		run := func(channels int) [][]float32 {
			cfg := DefaultConfig()
			cfg.Filter.ExportLinearAecOutput = true
			ec, err := New(cfg, rate, 1, channels)
			if err != nil {
				t.Fatal(err)
			}
			render := ec3Frame(1, n)
			capture := ec3Frame(channels, n)
			linear := ec3Frame(channels, LinearOutputFrameSize)
			out := make([][]float32, channels)
			var state uint32 = 1
			noise := func() float32 {
				state = state*1664525 + 1013904223
				return float32(int32(state>>8)%4096-2048) / 8192
			}
			history := make([]float32, 0, frames*n)
			for f := 0; f < frames; f++ {
				for i := range render[0] {
					render[0][i] = noise()
				}
				history = append(history, render[0]...)
				for i := range capture[0] {
					var y float32
					if k := len(history) - n + i - rate/50; k >= 0 {
						y = 0.5 * history[k]
					}
					// Nearend speech in the second half.
					if f >= frames/2 {
						y += 0.1 * noise()
					}
					for ch := range capture {
						capture[ch][i] = y
					}
				}
				if err := ec.AnalyzeRender(render); err != nil {
					t.Fatal(err)
				}
				if err := ec.ProcessCaptureWithLinearOutput(capture, linear); err != nil {
					t.Fatal(err)
				}
				for ch := range capture {
					out[ch] = append(out[ch], linear[ch]...)
				}
			}
			return out
		}
		want := run(1)[0]
		for channels := 2; channels <= 4; channels++ {
			got := run(channels)[0]
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%d Hz, %d capture channels: the linear output of the first channel differs from that of one channel at sample %d",
						rate, channels, i)
				}
			}
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

// gcMarkCPU returns the CPU time the garbage collector has spent marking.
func gcMarkCPU() float64 {
	samples := []metrics.Sample{
		{Name: "/cpu/classes/gc/mark/assist:cpu-seconds"},
		{Name: "/cpu/classes/gc/mark/dedicated:cpu-seconds"},
		{Name: "/cpu/classes/gc/mark/idle:cpu-seconds"},
	}
	metrics.Read(samples)
	var seconds float64
	for _, s := range samples {
		seconds += s.Value.Float64()
	}
	return seconds
}

// gcCycles runs n garbage collection cycles and returns the shortest and
// the least mark CPU time of one, in nanoseconds.
func gcCycles(n int) (cycle, mark float64) {
	cycle, mark = math.Inf(1), math.Inf(1)
	for i := 0; i < n; i++ {
		m, start := gcMarkCPU(), time.Now()
		runtime.GC()
		cycle = min(cycle, float64(time.Since(start)))
		mark = min(mark, (gcMarkCPU()-m)*1e9)
	}
	return cycle, mark
}

// newCancellers returns n cancellers of one footprint case.
func newCancellers(t testing.TB, n, rate, ch int) []*EchoCanceller {
	live := make([]*EchoCanceller, n)
	for i := range live {
		ec, err := New(DefaultConfig(), rate, ch, ch)
		if err != nil {
			t.Fatal(err)
		}
		live[i] = ec
	}
	return live
}

// TestGCMarkPerCanceller bounds what a live canceller adds to the mark phase
// of a garbage collection cycle on one P: the mark CPU time of a cycle with
// 2000 cancellers less that of a cycle without them. The bound is several
// times the cost on a current machine; a canceller whose state the garbage
// collector had to read cost ten times as much.
func TestGCMarkPerCanceller(t *testing.T) {
	if testing.Short() || raceEnabled {
		t.Skip("timing")
	}
	const n, maxNs = 2000, 1500
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	for _, c := range footprintCases {
		_, empty := gcCycles(20)
		live := newCancellers(t, n, c.rate, c.ch)
		_, mark := gcCycles(20)
		runtime.KeepAlive(live)
		net := (mark - empty) / n
		t.Logf("%s: %.0f ns of mark per canceller and cycle", c.name, net)
		if net > maxNs {
			t.Errorf("%s: %.0f ns of mark per canceller and cycle, want at most %d", c.name, net, maxNs)
		}
	}
}

// BenchmarkGCMark measures one full GC cycle with 1000 live cancellers. With
// -cpu 1 the time is close to the mark work; with more Ps it is mostly the
// coordination of the mark workers. mark-ns/op is the CPU time of the mark
// phase alone, which leaves out what the cycle costs whatever the heap holds
// (stopping and starting the world, the sweep). A cycle costs time without
// any canceller too: net-ns/op and net-mark-ns/op are the two less the
// shortest of 50 cycles run before the cancellers exist, which is what the
// cancellers add.
func BenchmarkGCMark(b *testing.B) {
	for _, c := range footprintCases {
		b.Run(fmt.Sprintf("%s_x1000", c.name), func(b *testing.B) {
			emptyCycle, emptyMark := gcCycles(50)
			live := newCancellers(b, 1000, c.rate, c.ch)
			runtime.GC()
			mark := gcMarkCPU()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runtime.GC()
			}
			b.StopTimer()
			markNs := (gcMarkCPU() - mark) * 1e9 / float64(b.N)
			b.ReportMetric(markNs, "mark-ns/op")
			b.ReportMetric(markNs-emptyMark, "net-mark-ns/op")
			b.ReportMetric(float64(b.Elapsed())/float64(b.N)-emptyCycle, "net-ns/op")
			runtime.KeepAlive(live)
		})
	}
}
