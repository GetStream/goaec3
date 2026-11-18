package goaec3

import (
	"fmt"
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

// TestNewFootprint reports the live heap per canceller and bounds the number
// of heap objects, which sets the GC mark cost of many live cancellers.
func TestNewFootprint(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	for _, c := range footprintCases {
		bytes, objects := footprint(t, c.rate, c.ch)
		t.Logf("%s: %d B, %d objects per canceller", c.name, bytes, objects)
		if objects > 250 {
			t.Errorf("%s: %d heap objects per canceller, want at most 250", c.name, objects)
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

// BenchmarkGCMark measures one full GC cycle with 1000 live cancellers.
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
