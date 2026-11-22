package aec3

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestDlyCoreScalarMatchesMatchedFilterCore checks dlyCoreScalar with one to
// three filters in step, with and without the accumulated error on the
// first, against MatchedFilterCore on each filter alone. The windows wrap at
// different places, samples update or not, capture samples saturate, values
// are zero or tiny, and calls are longer than one run of x*x sums.
func TestDlyCoreScalarMatchesMatchedFilterCore(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	trials := 6000
	if testing.Short() {
		trials = 1200
	}
	for trial := 0; trial < trials; trial++ {
		c := dlyRandCoreTrial(rng, trial, 4)
		hSize := len(c.h)
		if len(c.x) < hSize+dlyX2Run {
			c.x = dlyRandSlice(rng, hSize+dlyX2Run+rng.Intn(40), 3000)
			c.xs = rng.Intn(len(c.x))
		}
		if trial%7 == 0 {
			// Two runs of x*x sums.
			c.y = dlyRandSlice(rng, dlyX2Run+1+rng.Intn(dlyX2Run), 3000)
		}
		if trial%13 == 3 {
			// x*x sums equal to the threshold: no update.
			for k := range c.x {
				c.x[k] = float32(2*rng.Intn(2) - 1)
			}
			c.thr = float32(hSize)
		}
		xSize := len(c.x)
		nf := 1 + rng.Intn(dlyScalarMax)
		acc := trial%2 == 0
		var fs [dlyScalarMax]dlyScalarFilter
		var hWant [dlyScalarMax][]float32
		var xs [dlyScalarMax]int
		var es0 [dlyScalarMax]float32
		for i := 0; i < nf; i++ {
			h := c.h
			if i > 0 {
				h = dlyRandSlice(rng, hSize, 0.01)
				if trial%11 == 0 {
					for k := range h {
						h[k] = float32(rng.Intn(5)-2) / 4
					}
				}
			}
			switch rng.Intn(4) {
			case 0:
				xs[i] = c.xs
			case 1:
				// Next to the first filter's wrap.
				xs[i] = (c.xs + rng.Intn(5)) % xSize
			case 2:
				xs[i] = (c.xs + 3*hSize/4) % xSize
			default:
				xs[i] = rng.Intn(xSize)
			}
			if i == 0 {
				xs[i] = c.xs
			}
			es0[i] = float32(rng.NormFloat64())
			hWant[i] = append([]float32(nil), h...)
			fs[i] = dlyScalarFilter{h: append([]float32(nil), h...), xStart: xs[i], errorSum: es0[i]}
		}
		accGot := make([]float32, hSize/4)
		if acc {
			fs[0].accErr = accGot
		}
		scratch := make([]float32, dlyScalarScratch(hSize))
		dlyCoreScalar(fs[:nf], c.thr, 0.7, c.x, c.y, scratch, false)
		for i := 0; i < nf; i++ {
			var updWant bool
			esWant := es0[i]
			accWant := dlyRandSlice(rng, hSize/4, 1) // MatchedFilterCore zeroes it
			MatchedFilterCore(xs[i], c.thr, 0.7, c.x, c.y, hWant[i], &updWant, &esWant, acc && i == 0, accWant)
			what := fmt.Sprintf("trial %d filter %d of %d", trial, i, nf)
			dlyRequireBits(t, what+" h", fs[i].h, hWant[i])
			if fs[i].updated != updWant || math.Float32bits(fs[i].errorSum) != math.Float32bits(esWant) {
				t.Fatalf("%s: updated %v/%v errorSum %v/%v", what, fs[i].updated, updWant, fs[i].errorSum, esWant)
			}
			if acc && i == 0 {
				dlyRequireBits(t, what+" accumulated error", accGot, accWant)
			}
			wantStart := ((xs[i]-len(c.y))%xSize + xSize) % xSize
			if fs[i].xStart != wantStart || fs[i].pending {
				t.Fatalf("%s: xStart %d, want %d; pending %v", what, fs[i].xStart, wantStart, fs[i].pending)
			}
		}
	}
}

// TestDlyMatchedFilterScalarInStepMatchesSingle checks that a scalar-lane
// MatchedFilter that runs its filters in step and one that runs
// MatchedFilterCore have the same state after every block.
func TestDlyMatchedFilterScalarInStepMatchesSingle(t *testing.T) {
	saved := dlyScalarInStep
	defer func() { dlyScalarInStep = saved }()
	numBlocks := 1200
	if testing.Short() {
		numBlocks = 300
	}
	for _, numFilters := range []int{1, 2, 4, 5, 7} {
		for _, factor := range []int{4, 8} {
			for _, preEcho := range []bool{false, true} {
				limit := float32(150)
				if factor == 8 {
					limit = 20
				}
				sub := BlockSize / factor
				newFilter := func() *MatchedFilter {
					return NewMatchedFilter(OptNone, sub, MatchedFilterWindowSizeSubBlocks, numFilters,
						MatchedFilterAlignmentShiftSizeSubBlocks, limit, 0.7, 0.3, 0.2, preEcho)
				}
				inStep, single := newFilter(), newFilter()
				s := newDlyQuietScene(int64(900+numFilters+factor), factor, numFilters,
					math.Float32frombits(inStep.quietLimitBits))
				reported := 0
				for blk := 0; blk < numBlocks; blk++ {
					mode := dlyQuietLoud
					if blk%400 > 300 {
						mode = dlyQuietBarelyAbove
					}
					y := s.block(mode, limit)
					if s.rng.Intn(40) == 0 {
						y[s.rng.Intn(len(y))] = 32700
					}
					slow := s.rng.Intn(3) == 0
					dlyScalarInStep = true
					inStep.Update(s.buf, y, slow)
					dlyScalarInStep = false
					single.Update(s.buf, y, slow)
					what := fmt.Sprintf("%d filters ds%d preEcho %v block %d", numFilters, factor, preEcho, blk)
					if d := dlyQuietStateDiff(inStep, single); d != "" {
						t.Fatalf("%s: %s", what, d)
					}
					if preEcho {
						dlyRequireBits(t, what+" instantaneous accumulated error",
							inStep.instantaneousAccumulatedError(), single.instantaneousAccumulatedError())
					}
					if single.hasReportedLagEstimate {
						reported++
					}
					if s.rng.Intn(500) == 0 {
						inStep.Reset(false)
						single.Reset(false)
					}
				}
				if numFilters >= 4 && reported < numBlocks/10 {
					t.Errorf("%d filters ds%d preEcho %v: only %d lag reports", numFilters, factor, preEcho, reported)
				}
			}
		}
	}
}

// TestDlyMatchedFilterUpdateAllocationFree checks that Update does not
// allocate in any lane.
func TestDlyMatchedFilterUpdateAllocationFree(t *testing.T) {
	for _, opt := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		for _, preEcho := range []bool{false, true} {
			m := NewMatchedFilter(opt, 16, MatchedFilterWindowSizeSubBlocks, 5,
				MatchedFilterAlignmentShiftSizeSubBlocks, 150, 0.7, 0.3, 0.2, preEcho)
			s := newDlyQuietScene(int64(50+int(opt)), 4, 5, math.Float32frombits(m.quietLimitBits))
			var y []float32
			for i := 0; i < 300; i++ {
				y = s.block(dlyQuietLoud, 150)
				m.Update(s.buf, y, false)
			}
			if a := testing.AllocsPerRun(20, func() { m.Update(s.buf, y, false) }); a != 0 {
				t.Errorf("opt %d preEcho %v: %v allocations per Update", opt, preEcho, a)
			}
		}
	}
}
