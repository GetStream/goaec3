//go:build webrtc_oracle

package webrtcoracle

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/GetStream/goaec3/internal/aec3"
)

func TestDlyAlignmentMixerParity(t *testing.T) {
	type mixCfg struct {
		downmix, adaptive, prefer bool
	}
	for _, ch := range []int{1, 2, 3, 4} {
		for _, mc := range []mixCfg{{true, false, false}, {false, true, true}, {false, true, false}, {false, false, false}} {
			t.Run(fmt.Sprintf("ch%d_%+v", ch, mc), func(t *testing.T) {
				rng := rand.New(rand.NewSource(18))
				g := aec3.NewAlignmentMixerWithParams(ch, mc.downmix, mc.adaptive, 10000, mc.prefer)
				c := NewDlyAlignmentMixer(ch, mc.downmix, mc.adaptive, 10000, mc.prefer)
				defer c.Free()
				// Runs past the 60 s energy-smoothing switchover.
				const numBlocks = 16000
				flat := make([]float32, ch*aec3.BlockSize)
				blk := aec3.NewBlock(1, ch)
				yg := make([]float32, aec3.BlockSize)
				yc := make([]float32, aec3.BlockSize)
				for k := 0; k < numBlocks; k++ {
					for cc := 0; cc < ch; cc++ {
						// The last channel dominates first; channel 0 turns
						// strong later and channel 1 takes over after the
						// smoothing switchover.
						amp := 40.0
						switch {
						case cc == ch-1 && k < 400:
							amp = 6000
						case cc == 0 && k >= 300 && k < 15500:
							amp = 3000
						case cc == 1 && k >= 15500:
							amp = 5000
						case cc == ch-1:
							amp = 1500
						}
						v := blk.View(0, cc)
						for i := range v {
							x := float32(rng.NormFloat64() * amp)
							v[i] = x
							flat[cc*aec3.BlockSize+i] = x
						}
					}
					g.ProduceOutput(&blk, yg)
					c.ProduceOutput(flat, yc)
					if i, ok := dlyEqualBits(yg, yc); !ok {
						t.Fatalf("block %d [%d]: got %v want %v", k, i, yg[i], yc[i])
					}
				}
			})
		}
	}
}

func TestDlyClockdriftDetectorParity(t *testing.T) {
	rng := rand.New(rand.NewSource(19))
	g := aec3.NewClockdriftDetector()
	c := NewDlyClockdriftDetector()
	defer c.Free()
	var seq []int
	x := 100
	for r := 0; r < 40; r++ {
		switch rng.Intn(5) {
		case 0: // drift up: x-3, x-2, x-1, x
			seq = append(seq, x+1, x+2, x+3)
			x += 3
		case 1: // drift down with swapped order
			seq = append(seq, x-2, x-1, x-3)
			x -= 3
		case 2: // probable drift only
			seq = append(seq, x+2, x+1)
		case 3: // long stable run resetting the level
			for i := 0; i < 7600; i++ {
				seq = append(seq, x)
			}
		default:
			for i := 0; i < 5; i++ {
				seq = append(seq, x+rng.Intn(7)-3)
			}
		}
	}
	levels := map[int]int{}
	for i, d := range seq {
		g.Update(d)
		lc := c.Update(d)
		levels[lc]++
		if int(g.ClockdriftLevel()) != lc {
			t.Fatalf("update %d (%d): level %d want %d", i, d, g.ClockdriftLevel(), lc)
		}
	}
	if len(levels) < 3 {
		t.Fatalf("sequence did not visit all levels: %v", levels)
	}
}
