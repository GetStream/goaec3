package aec3

import (
	"math/rand"
	"testing"
)

func dlyFillBlock(rng *rand.Rand, b *Block, amp float64) {
	for band := 0; band < b.NumBands(); band++ {
		for ch := 0; ch < b.NumChannels(); ch++ {
			v := b.View(band, ch)
			for i := range v {
				v[i] = float32(rng.NormFloat64() * amp)
			}
		}
	}
}

// TestDlyDelayPathAllocationFree locks the steady-state render/delay path
// (RenderDelayBuffer, RenderDelayController and with them MatchedFilter,
// EchoPathDelayEstimator and the decimators) to zero allocations.
func TestDlyDelayPathAllocationFree(t *testing.T) {
	for _, preEcho := range []bool{false, true} {
		cfg := DefaultEchoCanceller3Config()
		cfg.Delay.DetectPreEcho = preEcho
		rng := rand.New(rand.NewSource(1))
		rdb := NewRenderDelayBuffer(&cfg, 48000, 2)
		rdc := NewRenderDelayController(&cfg, 48000, 2)
		render := NewBlock(3, 2)
		capture := NewBlock(3, 2)
		step := func() {
			dlyFillBlock(rng, &render, 3000)
			dlyFillBlock(rng, &capture, 3000)
			rdb.Insert(&render)
			rdb.PrepareCaptureProcessing()
			if d, ok := rdc.GetDelay(rdb.GetDownsampledRenderBuffer(), rdb.Delay(), &capture); ok {
				rdb.AlignFromDelay(d.Delay)
			}
		}
		for i := 0; i < 50; i++ {
			step()
		}
		if a := testing.AllocsPerRun(200, step); a != 0 {
			t.Fatalf("preEcho=%v: %v allocations per block", preEcho, a)
		}
	}
}

func TestDlyFramingAllocationFree(t *testing.T) {
	blocker := NewFrameBlocker(3, 2)
	framer := NewBlockFramer(3, 2)
	delay := NewBlockDelayBuffer(2, 3, 160, 17)
	frame := NewFrame(3, 2)
	block := NewBlock(3, 2)
	step := func() {
		delay.DelaySignal(frame)
		for i := 0; i < 2; i++ {
			blocker.InsertSubFrameAndExtractBlock(frame, i, &block)
			framer.InsertBlockAndExtractSubFrame(&block, frame, i)
		}
		if blocker.IsBlockAvailable() {
			blocker.ExtractBlock(&block)
			framer.InsertBlock(&block)
		}
	}
	for i := 0; i < 10; i++ {
		step()
	}
	if a := testing.AllocsPerRun(200, step); a != 0 {
		t.Fatalf("%v allocations per frame", a)
	}
}

func BenchmarkRenderDelayControllerGetDelay(b *testing.B) {
	cfg := DefaultEchoCanceller3Config()
	rng := rand.New(rand.NewSource(2))
	rdb := NewRenderDelayBuffer(&cfg, 16000, 1)
	rdc := NewRenderDelayController(&cfg, 16000, 1)
	render := NewBlock(1, 1)
	capture := NewBlock(1, 1)
	dlyFillBlock(rng, &render, 3000)
	dlyFillBlock(rng, &capture, 3000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rdb.Insert(&render)
		rdb.PrepareCaptureProcessing()
		rdc.GetDelay(rdb.GetDownsampledRenderBuffer(), rdb.Delay(), &capture)
	}
}
