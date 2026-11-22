package goaec3

import (
	"errors"
	"math"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
)

// TestResourceLimits: the fields that size a canceller are accepted at
// their limits and rejected above them, in either config, before New
// allocates anything (Filter.Refined.LengthBlocks = MaxInt32 asks for
// hundreds of gigabytes).
func TestResourceLimits(t *testing.T) {
	fields := map[string]struct {
		limit int
		set   func(c *Config, v int)
	}{
		"Filter.Refined.LengthBlocks": {MaxFilterLengthBlocks, func(c *Config, v int) { c.Filter.Refined.LengthBlocks = v }},
		"Filter.Coarse.LengthBlocks":  {MaxFilterLengthBlocks, func(c *Config, v int) { c.Filter.Coarse.LengthBlocks = v }},
		"Delay.NumFilters":            {MaxDelayFilters, func(c *Config, v int) { c.Delay.NumFilters = v }},
	}
	for name, f := range fields {
		cfg := DefaultConfig()
		f.set(&cfg, f.limit)
		if _, err := New(cfg, 16000, 1, 1); err != nil {
			t.Errorf("%s = %d: %v", name, f.limit, err)
		}
		for _, v := range []int{f.limit + 1, 5000, math.MaxInt32, math.MaxInt} {
			cfg := DefaultConfig()
			f.set(&cfg, v)
			if _, err := New(cfg, 16000, 1, 1); !errors.Is(err, ErrConfig) {
				t.Errorf("%s = %d: New error = %v, want ErrConfig", name, v, err)
			}
			mc := MultichannelConfig()
			f.set(&mc, v)
			if _, err := NewWithMultichannelConfig(DefaultConfig(), mc, 16000, 2, 1); !errors.Is(err, ErrConfig) {
				t.Errorf("%s = %d (multichannel config): error = %v, want ErrConfig", name, v, err)
			}
		}
	}
	for _, ch := range [][2]int{{MaxChannels + 1, 1}, {1, MaxChannels + 1}, {math.MaxInt32, 1}, {1, math.MaxInt}} {
		if _, err := New(DefaultConfig(), 48000, ch[0], ch[1]); !errors.Is(err, ErrNumChannels) {
			t.Errorf("channels %v: New error = %v, want ErrNumChannels", ch, err)
		}
	}
	if !strings.Contains(ErrNumChannels.Error(), strconv.Itoa(MaxChannels)) {
		t.Errorf("ErrNumChannels %q does not name MaxChannels = %d", ErrNumChannels, MaxChannels)
	}
}

// robustLargestConfig returns the config that makes New allocate the most:
// every field that sizes a buffer at its limit.
func robustLargestConfig() Config {
	cfg := DefaultConfig()
	cfg.Delay.NumFilters = MaxDelayFilters
	cfg.Delay.FixedCaptureDelaySamples = 5000
	cfg.Filter.Refined.LengthBlocks = MaxFilterLengthBlocks
	cfg.Filter.Coarse.LengthBlocks = MaxFilterLengthBlocks
	cfg.Filter.RefinedInitial.LengthBlocks = MaxFilterLengthBlocks
	cfg.Filter.CoarseInitial.LengthBlocks = MaxFilterLengthBlocks
	cfg.Filter.ExportLinearAecOutput = true
	cfg.Erle.NumSections = MaxFilterLengthBlocks
	cfg.EchoModel.RenderPreWindowSize = 100
	cfg.EchoModel.RenderPostWindowSize = 100
	cfg.Suppressor.NearendAverageBlocks = 5000
	cfg.Suppressor.UseSubbandNearendDetection = true
	cfg.Suppressor.SubbandNearendDetection.NearendAverageBlocks = 1024
	return cfg
}

// TestLargestCanceller bounds the memory of the largest canceller New
// creates, the figure the package documentation gives, and runs it.
func TestLargestCanceller(t *testing.T) {
	const maxBytes = 64 << 20
	cfg := robustLargestConfig()
	if v := cfg; !v.Validate() {
		t.Fatal("the largest config is not valid")
	}
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	e, err := NewWithMultichannelConfig(cfg, cfg, 48000, MaxChannels, MaxChannels)
	if err != nil {
		t.Fatal(err)
	}
	n := e.FrameSize()
	scene := newRobustScene(1, 48000, 40)
	render, capture := robustFrame(MaxChannels, n), robustFrame(MaxChannels, n)
	linear := robustFrame(MaxChannels, LinearOutputFrameSize)
	frames := 30
	if testing.Short() {
		frames = 6
	}
	for range frames {
		scene.render(render, robustSigLoudNoise)
		scene.capture(capture, robustSigEcho)
		if err := e.AnalyzeRender(render); err != nil {
			t.Fatal(err)
		}
		if err := e.ProcessCaptureWithLinearOutput(capture, linear); err != nil {
			t.Fatal(err)
		}
		robustCheckFinite(t, "output", capture)
		robustCheckFinite(t, "linear output", linear)
	}
	runtime.ReadMemStats(&after)
	// TotalAlloc also counts the frames of the test, which are small.
	if got := after.TotalAlloc - before.TotalAlloc; got > maxBytes {
		t.Errorf("the largest canceller allocates %d MB, more than %d MB", got>>20, maxBytes>>20)
	} else {
		t.Logf("the largest canceller allocates %.1f MB", float64(got)/(1<<20))
	}
	runtime.KeepAlive(e)
}

// TestCallsBeforeAnyFrame calls every method that takes no frame on a new
// canceller, and a capture frame before any render frame.
func TestCallsBeforeAnyFrame(t *testing.T) {
	for _, external := range []bool{false, true} {
		cfg := DefaultConfig()
		cfg.Delay.UseExternalDelayEstimator = external
		e, err := New(cfg, 32000, 2, 2)
		if err != nil {
			t.Fatal(err)
		}
		if m := e.Metrics(); math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement) {
			t.Errorf("metrics before any frame: %+v", m)
		}
		for _, d := range []int{math.MinInt, -1, 0, math.MaxInt} {
			e.SetAudioBufferDelay(d)
		}
		e.SetCaptureOutputUsage(false)
		e.NotifyEchoPathGainChange()
		_ = e.Metrics()
		capture := robustFrame(2, e.FrameSize())
		for range 20 {
			if err := e.ProcessCapture(capture); err != nil {
				t.Fatal(err)
			}
			robustCheckFinite(t, "output", capture)
		}
	}
}

// TestErrorsLeaveTheStateUnchanged: a call that returns an error changes
// nothing. A canceller that gets malformed frames between its frames
// produces the output of one that does not, bit for bit.
func TestErrorsLeaveTheStateUnchanged(t *testing.T) {
	for _, rate := range []int{16000, 48000} {
		cfg := DefaultConfig()
		cfg.Filter.ExportLinearAecOutput = true
		var e [2]*EchoCanceller
		for i := range e {
			var err error
			if e[i], err = New(cfg, rate, 2, 2); err != nil {
				t.Fatal(err)
			}
		}
		n := e[0].FrameSize()
		scene := newRobustScene(5, rate, 30)
		render := robustFrame(2, n)
		var capture, linear [2][][]float32
		for i := range capture {
			capture[i], linear[i] = robustFrame(2, n), robustFrame(2, LinearOutputFrameSize)
		}
		short := robustFrame(2, n-1)
		for i := range short {
			for k := range short[i] {
				short[i][k] = 0.7
			}
		}
		for frame := range 150 {
			scene.render(render, robustSigEcho)
			scene.capture(capture[0], robustSigEcho)
			for ch := range capture[0] {
				copy(capture[1][ch], capture[0][ch])
			}
			for _, err := range []error{
				e[1].AnalyzeRender(short),
				e[1].AnalyzeRender(render[:1]),
				e[1].ProcessCapture(short),
				e[1].ProcessCaptureWithLinearOutput(capture[1], short),
				e[1].ProcessCaptureWithLinearOutput(short, linear[1]),
			} {
				if err == nil {
					t.Fatal("a malformed frame returned no error")
				}
			}
			for i := range e {
				if err := e[i].AnalyzeRender(render); err != nil {
					t.Fatal(err)
				}
				if err := e[i].ProcessCaptureWithLinearOutput(capture[i], linear[i]); err != nil {
					t.Fatal(err)
				}
			}
			for ch := range capture[0] {
				for k := range capture[0][ch] {
					if math.Float32bits(capture[0][ch][k]) != math.Float32bits(capture[1][ch][k]) {
						t.Fatalf("%d Hz frame %d ch %d sample %d: %v, without the errors %v", rate, frame, ch, k,
							capture[1][ch][k], capture[0][ch][k])
					}
				}
				for k := range linear[0][ch] {
					if math.Float32bits(linear[0][ch][k]) != math.Float32bits(linear[1][ch][k]) {
						t.Fatalf("%d Hz frame %d ch %d linear sample %d differs", rate, frame, ch, k)
					}
				}
			}
			if e[0].Metrics() != e[1].Metrics() {
				t.Fatalf("%d Hz frame %d: metrics %+v, without the errors %+v", rate, frame, e[1].Metrics(),
					e[0].Metrics())
			}
		}
	}
}

// TestAliasedBuffers: AnalyzeRender copies its frame, so the caller may
// process the capture signal in the same buffers; the output is that of
// separate buffers. One buffer passed as two capture channels, or as a
// capture channel and its linear output, holds the channel written last.
func TestAliasedBuffers(t *testing.T) {
	for _, rate := range []int{16000, 48000} {
		cfg := DefaultConfig()
		cfg.Filter.ExportLinearAecOutput = true
		var e [3]*EchoCanceller
		for i := range e {
			var err error
			if e[i], err = New(cfg, rate, 2, 2); err != nil {
				t.Fatal(err)
			}
		}
		n := e[0].FrameSize()
		scene := newRobustScene(9, rate, 30)
		render, capture, shared := robustFrame(2, n), robustFrame(2, n), robustFrame(2, n)
		linear := robustFrame(2, LinearOutputFrameSize)
		for frame := range 100 {
			scene.render(render, robustSigEcho)
			scene.capture(capture, robustSigEcho)
			// e[0]: separate buffers. e[1]: the capture signal overwrites
			// the render frame after AnalyzeRender.
			for ch := range shared {
				copy(shared[ch], render[ch])
			}
			if err := e[0].AnalyzeRender(render); err != nil {
				t.Fatal(err)
			}
			if err := e[1].AnalyzeRender(shared); err != nil {
				t.Fatal(err)
			}
			for ch := range shared {
				copy(shared[ch], capture[ch])
			}
			if err := e[1].ProcessCapture(shared); err != nil {
				t.Fatal(err)
			}
			// e[2]: one buffer for both capture channels, then for a
			// capture channel and a linear output channel.
			same := [][]float32{render[0], render[0]}
			if err := e[2].AnalyzeRender(same); err != nil {
				t.Fatal(err)
			}
			if err := e[2].ProcessCapture(same); err != nil {
				t.Fatal(err)
			}
			robustCheckFinite(t, "one buffer as two channels", same)
			if rate == 16000 {
				if err := e[2].ProcessCaptureWithLinearOutput(linear, linear); err != nil {
					t.Fatal(err)
				}
				robustCheckFinite(t, "capture and linear output in one buffer", linear)
			}
			if err := e[0].ProcessCapture(capture); err != nil {
				t.Fatal(err)
			}
			for ch := range capture {
				for k := range capture[ch] {
					if math.Float32bits(capture[ch][k]) != math.Float32bits(shared[ch][k]) {
						t.Fatalf("%d Hz frame %d ch %d sample %d: %v in the render buffers, %v in separate ones",
							rate, frame, ch, k, shared[ch][k], capture[ch][k])
					}
				}
			}
		}
	}
}

// TestConcurrentContractStress drives one canceller the way the contract
// allows, under load: AnalyzeRender on one goroutine, the capture-side
// methods on another, NotifyEchoPathGainChange and the accessors on a third,
// with neither side waiting for the other, so the render queue runs empty
// and overflows. Run with -race.
func TestConcurrentContractStress(t *testing.T) {
	frames := 3000
	if testing.Short() {
		frames = 300
	}
	cfg := DefaultConfig()
	cfg.Filter.ExportLinearAecOutput = true
	mc := MultichannelConfig()
	robustAgree(&mc, &cfg)
	e, err := NewWithMultichannelConfig(cfg, mc, 48000, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	n := e.FrameSize()
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(2)
	go func() {
		defer wg.Done()
		scene := newRobustScene(1, 48000, 30)
		render := robustFrame(2, n)
		for i := range frames {
			// Mono and stereo content in turns, so the capture side sets
			// its block processor up again while frames arrive.
			scene.render(render, robustSigEcho)
			if i/200%2 == 0 {
				copy(render[1], render[0])
			}
			if err := e.AnalyzeRender(render); err != nil {
				t.Error(err)
				return
			}
			if i%64 == 0 {
				runtime.Gosched()
			}
		}
	}()
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			e.NotifyEchoPathGainChange()
			if e.SampleRateHz() != 48000 || e.FrameSize() != n || e.NumRenderChannels() != 2 ||
				e.NumCaptureChannels() != 2 {
				t.Error("an accessor changed its value")
				return
			}
			runtime.Gosched()
		}
	}()
	scene := newRobustScene(2, 48000, 30)
	capture := robustFrame(2, n)
	linear := robustFrame(2, LinearOutputFrameSize)
	for i := range frames {
		scene.render(robustFrame(1, n), robustSigEcho)
		scene.capture(capture, robustSigEcho)
		switch i % 7 {
		case 1:
			e.SetAudioBufferDelay(i - frames/2)
		case 3:
			e.SetCaptureOutputUsage(i%14 == 3)
		}
		if i%2 == 0 {
			err = e.ProcessCapture(capture)
		} else {
			err = e.ProcessCaptureWithLinearOutput(capture, linear)
		}
		if err != nil {
			t.Fatal(err)
		}
		robustCheckFinite(t, "output", capture)
		if m := e.Metrics(); math.IsNaN(m.EchoReturnLoss) || math.IsNaN(m.EchoReturnLossEnhancement) {
			t.Fatalf("metrics %+v", m)
		}
	}
	close(stop)
	wg.Wait()
}
