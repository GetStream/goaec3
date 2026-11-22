package aec3

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// dlyQuietScene feeds two MatchedFilters, with and without the quiet-render
// skip, the same render buffer and capture block by block. The render level
// follows a mode per block. The capture is the render delayed by a fixed lag
// plus noise, so the filters converge, report lags and run the pre-echo
// logic while the render is loud.
type dlyQuietScene struct {
	rng     *rand.Rand
	sub     int
	buf     *DownsampledRenderBuffer
	history []float32 // render samples in time order
	lag     int
	limit   float32 // quiet limit of the filter under test
}

func newDlyQuietScene(seed int64, factor, numFilters int, limit float32) *dlyQuietScene {
	s := &dlyQuietScene{rng: rand.New(rand.NewSource(seed)), sub: BlockSize / factor, limit: limit}
	s.buf = NewDownsampledRenderBuffer(GetDownSampledBufferSize(factor, numFilters))
	s.buf.Read = s.buf.OffsetIndex(s.buf.Write, s.sub)
	s.lag = 40 + s.rng.Intn(300)
	return s
}

const (
	dlyQuietLoud = iota
	dlyQuietZero
	dlyQuietAtLimit     // every sample is +-limit exactly
	dlyQuietBelowLimit  // random magnitudes up to the limit
	dlyQuietAboveLimit  // every sample is +-nextafter(limit)
	dlyQuietBarelyAbove // constant magnitude a little over sqrt(threshold/N)
	dlyQuietSparseLoud  // quiet with one loud sample per block
	dlyQuietNaN         // quiet with a NaN or an infinity
	dlyQuietModes
)

// block writes the next render sub-block to the buffer as
// RenderDelayBufferImpl does, backwards and reversed, and returns the
// capture.
func (s *dlyQuietScene) block(mode int, excitationLimit float32) []float32 {
	x := make([]float32, s.sub)
	sign := func() float32 {
		if s.rng.Intn(2) == 0 {
			return -1
		}
		return 1
	}
	for i := range x {
		switch mode {
		case dlyQuietLoud:
			x[i] = float32(s.rng.NormFloat64() * 3000)
		case dlyQuietZero:
			x[i] = 0
			if s.rng.Intn(4) == 0 {
				x[i] = float32(math.Copysign(0, -1))
			}
		case dlyQuietAtLimit:
			x[i] = sign() * s.limit
		case dlyQuietBelowLimit, dlyQuietSparseLoud, dlyQuietNaN:
			x[i] = sign() * s.limit * float32(s.rng.Float64())
		case dlyQuietAboveLimit:
			x[i] = sign() * math.Nextafter32(s.limit, float32(math.Inf(1)))
		case dlyQuietBarelyAbove:
			x[i] = sign() * excitationLimit * 1.001
		}
	}
	switch mode {
	case dlyQuietSparseLoud:
		x[s.rng.Intn(len(x))] = sign() * (s.limit*1.5 + 1)
	case dlyQuietNaN:
		v := float32(math.NaN())
		if s.rng.Intn(2) == 0 {
			v = float32(math.Inf(s.rng.Intn(2)*2 - 1))
		}
		x[s.rng.Intn(len(x))] = v
	}
	s.history = append(s.history, x...)
	s.buf.UpdateWriteIndex(-s.sub)
	for i := range x {
		s.buf.Buffer[s.buf.Write+i] = x[s.sub-1-i]
	}
	s.buf.UpdateReadIndex(-s.sub)

	y := make([]float32, s.sub)
	base := len(s.history) - s.sub - s.lag
	for i := range y {
		if k := base + i; k >= 0 {
			y[i] = 0.6 * s.history[k]
		}
		y[i] += float32(s.rng.NormFloat64() * 20)
	}
	if len(s.history) > 1<<16 {
		s.history = append(s.history[:0], s.history[len(s.history)-4096:]...)
	}
	return y
}

// dlyQuietStateDiff compares, bit for bit, every field of the two filters
// that carries state from one Update to the next or that Update reports. It
// leaves out instantaneousAccumulatedError and scratchMemory, which are
// scratch.
func dlyQuietStateDiff(a, b *MatchedFilter) string {
	for n := 0; n < a.numFilters; n++ {
		fa, fb := a.filter(n), b.filter(n)
		for k := range fa {
			if math.Float32bits(fa[k]) != math.Float32bits(fb[k]) {
				return fmt.Sprintf("filters[%d][%d]: %v vs %v", n, k, fa[k], fb[k])
			}
		}
		if !a.detectPreEcho {
			continue
		}
		ea, eb := a.accumulatedError(n), b.accumulatedError(n)
		for k := range ea {
			if math.Float32bits(ea[k]) != math.Float32bits(eb[k]) {
				return fmt.Sprintf("accumulatedError[%d][%d]: %v vs %v", n, k, ea[k], eb[k])
			}
		}
	}
	switch {
	case a.hasReportedLagEstimate != b.hasReportedLagEstimate || a.reportedLagEstimate != b.reportedLagEstimate:
		return fmt.Sprintf("reportedLagEstimate: %v/%v vs %v/%v", a.reportedLagEstimate, a.hasReportedLagEstimate,
			b.reportedLagEstimate, b.hasReportedLagEstimate)
	case a.hasWinnerLag != b.hasWinnerLag || a.winnerLag != b.winnerLag:
		return fmt.Sprintf("winnerLag: %v/%v vs %v/%v", a.winnerLag, a.hasWinnerLag, b.winnerLag, b.hasWinnerLag)
	case a.lastDetectedBestLagFilter != b.lastDetectedBestLagFilter:
		return fmt.Sprintf("lastDetectedBestLagFilter: %d vs %d", a.lastDetectedBestLagFilter,
			b.lastDetectedBestLagFilter)
	case a.numberPreEchoUpdates != b.numberPreEchoUpdates:
		return fmt.Sprintf("numberPreEchoUpdates: %d vs %d", a.numberPreEchoUpdates, b.numberPreEchoUpdates)
	}
	return ""
}

type dlyQuietCase struct {
	opt             Aec3Optimization
	factor          int
	preEcho         bool
	excitationLimit float32
}

func (c dlyQuietCase) String() string {
	return fmt.Sprintf("opt%d_ds%d_preecho%v_limit%g", c.opt, c.factor, c.preEcho, c.excitationLimit)
}

type dlyQuietStats struct {
	skipped, ran, reported, preEchoUpdates int
	filterSkips                            int // cores of single quiet filters not run
}

// dlyQuietRun drives a MatchedFilter with the skip and one without for
// numBlocks blocks and returns the first difference in their state, if any.
// tamper, if not nil, adjusts the filter with the skip first.
func dlyQuietRun(c dlyQuietCase, seed int64, numBlocks int, modes []int, tamper func(m *MatchedFilter)) (string, dlyQuietStats) {
	const numFilters = 5
	sub := BlockSize / c.factor
	skip := NewMatchedFilter(c.opt, sub, MatchedFilterWindowSizeSubBlocks, numFilters,
		MatchedFilterAlignmentShiftSizeSubBlocks, c.excitationLimit, 0.7, 0.3, 0.2, c.preEcho)
	ref := NewMatchedFilter(c.opt, sub, MatchedFilterWindowSizeSubBlocks, numFilters,
		MatchedFilterAlignmentShiftSizeSubBlocks, c.excitationLimit, 0.7, 0.3, 0.2, c.preEcho)
	ref.quietSkip = false
	if tamper != nil {
		tamper(skip)
	}
	s := newDlyQuietScene(seed, c.factor, numFilters, math.Float32frombits(skip.quietLimitBits))
	var st dlyQuietStats
	mode, left := dlyQuietLoud, 0
	preEchoUpdates := 0
	for blk := 0; blk < numBlocks; blk++ {
		if left == 0 {
			mode = modes[s.rng.Intn(len(modes))]
			// Segments shorter and longer than the render buffer: a quiet
			// stretch reaches the skip once it fills the buffer.
			left = 1 + s.rng.Intn(4*len(s.buf.Buffer)/sub)
			if s.rng.Intn(3) == 0 {
				left = 1 + s.rng.Intn(20)
			}
		}
		left--
		y := s.block(mode, c.excitationLimit)
		if s.rng.Intn(40) == 0 {
			y[s.rng.Intn(len(y))] = 32700
		}
		slow := s.rng.Intn(3) == 0
		wouldSkip := skip.quietSkip && skip.renderIsQuiet(s.buf.Buffer)
		skip.Update(s.buf, y, slow)
		ref.Update(s.buf, y, slow)
		if wouldSkip {
			st.skipped++
			// A skipped block leaves the scratch stale. Poison it, so that a
			// later read spreads into the compared state.
			for _, scratch := range [][]float32{skip.instantaneousAccumulatedError(), skip.scratchMemory()} {
				for k := range scratch {
					scratch[k] = float32(math.NaN())
				}
			}
		} else {
			st.ran++
		}
		if ref.hasReportedLagEstimate {
			st.reported++
		}
		if ref.numberPreEchoUpdates > preEchoUpdates {
			st.preEchoUpdates++
		}
		preEchoUpdates = ref.numberPreEchoUpdates
		if d := dlyQuietStateDiff(skip, ref); d != "" {
			return fmt.Sprintf("block %d (mode %d, skipped %v): %s", blk, mode, wouldSkip, d), st
		}
		switch s.rng.Intn(400) {
		case 0:
			skip.Reset(false)
			ref.Reset(false)
		case 1:
			skip.Reset(true)
			ref.Reset(true)
		case 2:
			// A realignment of the render buffer: the read position jumps.
			s.buf.Read = s.rng.Intn(len(s.buf.Buffer))
		}
	}
	st.filterSkips = int(skip.QuietFilterSkipsForTesting())
	if ref.QuietFilterSkipsForTesting() != 0 {
		return "the filter without the skip skipped a filter core", st
	}
	return "", st
}

func dlyQuietCases() []dlyQuietCase {
	var cases []dlyQuietCase
	for _, opt := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		for _, factor := range []int{4, 8} {
			for _, preEcho := range []bool{false, true} {
				limit := float32(150)
				if factor == 8 {
					limit = 20
				}
				cases = append(cases, dlyQuietCase{opt, factor, preEcho, limit})
			}
		}
		// Other excitation limits, with thresholds that are zero, tiny,
		// huge, infinite or NaN.
		for i, limit := range []float32{0, 1e-25, 1e-12, 0.37, 1, 37.5, 1024, 30000, 1e17, 3e19,
			float32(math.Inf(1)), float32(math.NaN()), -150} {
			cases = append(cases, dlyQuietCase{opt, 4 + 4*(i%2), i%3 != 0, limit})
		}
	}
	return cases
}

// TestDlyQuietSkipStateMatchesNoSkip checks that a MatchedFilter with the
// quiet-render skip and one without have the same state after every block,
// in every lane. The render is loud, silent, quiet, at the limit, just above
// it and not finite, with transitions, resets and realignments of the render
// buffer.
func TestDlyQuietSkipStateMatchesNoSkip(t *testing.T) {
	numBlocks := 6000
	if testing.Short() {
		numBlocks = 1500
	}
	modes := make([]int, 0, dlyQuietModes+2)
	for m := 0; m < dlyQuietModes; m++ {
		modes = append(modes, m)
	}
	modes = append(modes, dlyQuietLoud, dlyQuietLoud, dlyQuietZero, dlyQuietAtLimit, dlyQuietBelowLimit)
	var total dlyQuietStats
	for ci, c := range dlyQuietCases() {
		diff, st := dlyQuietRun(c, int64(100+ci), numBlocks, modes, nil)
		if diff != "" {
			t.Errorf("%v: %s", c, diff)
		}
		t.Logf("%v: skipped %d, ran %d (%d filter cores skipped), lag reports %d, pre-echo updates %d", c,
			st.skipped, st.ran, st.filterSkips, st.reported, st.preEchoUpdates)
		total.skipped += st.skipped
		total.ran += st.ran
		total.reported += st.reported
		total.preEchoUpdates += st.preEchoUpdates
		// The short run does not visit both sides in every scenario.
		if !testing.Short() && (c.excitationLimit == 150 || c.excitationLimit == 20) {
			if st.skipped < numBlocks/30 || st.ran < numBlocks/30 || st.reported < numBlocks/60 {
				t.Errorf("%v: scenario too one-sided: skipped %d, ran %d, lag reports %d", c, st.skipped, st.ran,
					st.reported)
			}
			if c.preEcho && st.preEchoUpdates == 0 {
				t.Errorf("%v: the pre-echo path never ran", c)
			}
			if st.filterSkips < numBlocks/30 {
				t.Errorf("%v: only %d filter cores skipped in blocks that ran", c, st.filterSkips)
			}
		}
	}
	t.Logf("total: skipped %d, ran %d, lag reports %d", total.skipped, total.ran, total.reported)
}

// TestDlyMatchedFilterPairedUpdateMatchesUnpaired checks that Update with
// the NEON-lane cores two at a time and one at a time leaves the same state
// after every block of a loud and quiet render.
func TestDlyMatchedFilterPairedUpdateMatchesUnpaired(t *testing.T) {
	savedPair := dlyPairNeon
	defer func() { dlyPairNeon = savedPair }()
	numBlocks := 3000
	if testing.Short() {
		numBlocks = 600
	}
	for _, opt := range []Aec3Optimization{OptNeon} {
		for _, factor := range []int{4, 8} {
			for _, preEcho := range []bool{false, true} {
				limit := float32(150)
				if factor == 8 {
					limit = 20
				}
				sub := BlockSize / factor
				newFilter := func() *MatchedFilter {
					return NewMatchedFilter(opt, sub, MatchedFilterWindowSizeSubBlocks, 5,
						MatchedFilterAlignmentShiftSizeSubBlocks, limit, 0.7, 0.3, 0.2, preEcho)
				}
				paired, single := newFilter(), newFilter()
				s := newDlyQuietScene(int64(300+int(opt)+factor), factor, 5, math.Float32frombits(paired.quietLimitBits))
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
					dlyPairNeon = true
					paired.Update(s.buf, y, slow)
					dlyPairNeon = false
					single.Update(s.buf, y, slow)
					if d := dlyQuietStateDiff(paired, single); d != "" {
						t.Fatalf("opt %d ds%d preEcho %v block %d: %s", opt, factor, preEcho, blk, d)
					}
					if single.hasReportedLagEstimate {
						reported++
					}
					if s.rng.Intn(500) == 0 {
						paired.Reset(false)
						single.Reset(false)
					}
				}
				if reported < numBlocks/10 {
					t.Errorf("opt %d ds%d preEcho %v: only %d lag reports", opt, factor, preEcho, reported)
				}
			}
		}
	}
}

// TestDlyQuietLimitIsTight checks that the quiet limit is within 0.1% of the
// excitation limit: a render buffer of the constant magnitude 1.001 times
// the excitation limit updates the filters and does not count as quiet.
func TestDlyQuietLimitIsTight(t *testing.T) {
	for _, c := range []struct {
		n     int
		limit float32
	}{{512, 150}, {256, 20}, {512, 1}, {256, 30000}, {512, 1e-6}} {
		thr := float32(float32(c.n)*c.limit) * c.limit
		m, ok := dlyQuietLimit(c.n, thr, dlyQuietMargin(c.n))
		if !ok || !(m <= c.limit) || !(m >= c.limit*0.999) {
			t.Errorf("n=%d limit=%g: quiet limit %g (ok %v)", c.n, c.limit, m, ok)
		}
	}
	for _, c := range []struct {
		n   int
		thr float32
	}{{512, float32(math.Inf(1))}, {512, float32(math.NaN())}, {512, -1}, {1 << 17, 100}, {0, 100}} {
		if _, ok := dlyQuietLimit(c.n, c.thr, dlyQuietMargin(c.n)); ok {
			t.Errorf("n=%d threshold=%g: the skip must be disabled", c.n, c.thr)
		}
	}
	if m, ok := dlyQuietLimit(512, 1e-30, dlyQuietMargin(512)); !ok || m != 0 {
		t.Errorf("tiny threshold: quiet limit %g (ok %v), want 0", m, ok)
	}
}

// TestDlyQuietSkipUnsafeMarginIsDetected checks the test above: with a
// margin that puts the quiet limit above sqrt(threshold/N), render at that
// limit updates the filters without the skip while those with it stand
// still, and the comparison of the states reports it in every lane.
func TestDlyQuietSkipUnsafeMarginIsDetected(t *testing.T) {
	for _, c := range dlyQuietCases() {
		if c.excitationLimit != 150 && c.excitationLimit != 20 {
			continue
		}
		n := BlockSize / c.factor * MatchedFilterWindowSizeSubBlocks
		thr := float32(float32(n)*c.excitationLimit) * c.excitationLimit
		for _, margin := range []float64{-0.002, -0.05, -0.5} {
			unsafe := func(m *MatchedFilter) {
				limit, ok := dlyQuietLimit(n, thr, margin)
				if !ok || !(limit > c.excitationLimit) {
					t.Fatalf("%v: margin %g gives limit %g", c, margin, limit)
				}
				m.quietLimitBits = math.Float32bits(limit)
			}
			diff, _ := dlyQuietRun(c, 7, 1500, []int{dlyQuietLoud, dlyQuietAtLimit, dlyQuietAtLimit}, unsafe)
			if diff == "" {
				t.Errorf("%v: an unsafe quiet margin of %g went undetected", c, margin)
			}
		}
		// With the real margin the states agree.
		if diff, st := dlyQuietRun(c, 7, 1500, []int{dlyQuietLoud, dlyQuietAtLimit, dlyQuietAtLimit}, nil); diff != "" ||
			st.skipped == 0 {
			t.Errorf("%v: real margin: diff %q, skipped %d", c, diff, st.skipped)
		}
	}
}

// TestDlyQuietRenderScan checks renderIsQuiet against a plain scan on
// buffers with one sample that is loud or not finite, at every position.
func TestDlyQuietRenderScan(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	m := NewMatchedFilter(OptNone, 16, 32, 5, 24, 150, 0.7, 0.3, 0.2, true)
	limit := math.Float32frombits(m.quietLimitBits)
	loud := []float32{math.Nextafter32(limit, 1e9), -math.Nextafter32(limit, 1e9), 1e30, float32(math.Inf(1)),
		float32(math.Inf(-1)), float32(math.NaN()), -float32(math.NaN())}
	for trial := 0; trial < 20000; trial++ {
		n := 1 + rng.Intn(100)
		x := make([]float32, n)
		for i := range x {
			x[i] = limit * float32(rng.Float64()*2-1)
			if rng.Intn(10) == 0 {
				x[i] = float32(math.Copysign(float64(limit), rng.Float64()-0.5))
			}
		}
		want := true
		if rng.Intn(3) != 0 {
			x[rng.Intn(n)] = loud[rng.Intn(len(loud))]
			want = false
		}
		m.loudHint = rng.Intn(120)
		if got := m.renderIsQuiet(x); got != want {
			t.Fatalf("trial %d: renderIsQuiet = %v, want %v (%v)", trial, got, want, x)
		}
		if !want {
			if v := x[m.loudHint]; math.Abs(float64(v)) <= float64(limit) {
				t.Fatalf("trial %d: loudHint %d points at a quiet sample %v", trial, m.loudHint, v)
			}
		}
	}
}

// BenchmarkMatchedFilterUpdateQuiet times Update on a render buffer below
// the quiet limit, with and without the skip.
func BenchmarkMatchedFilterUpdateQuiet(b *testing.B) {
	for _, skip := range []bool{true, false} {
		b.Run(fmt.Sprintf("skip=%v", skip), func(b *testing.B) {
			m := NewMatchedFilter(DetectOptimization(), 16, 32, 5, 24, 150, 0.7, 0.3, 0.2, true)
			m.quietSkip = skip
			s := newDlyQuietScene(1, 4, 5, math.Float32frombits(m.quietLimitBits))
			var y []float32
			for i := 0; i < 400; i++ {
				y = s.block(dlyQuietBelowLimit, 150)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				m.Update(s.buf, y, false)
			}
		})
	}
}

// TestDlyQuietFilterSkipRunsAccumulatedErrorFilter checks that the
// quiet-filter skip leaves the filter that computes the accumulated error
// alone. With the delay of the echo path in the overlap of two filters, the
// win of the older filter goes to the newer one, whose windows turn quiet
// first when the render falls silent. Its accumulated error of such a block
// still enters the pre-echo state.
func TestDlyQuietFilterSkipRunsAccumulatedErrorFilter(t *testing.T) {
	const numFilters = 5
	sub := BlockSize / 4
	numBlocks := 4000
	if testing.Short() {
		numBlocks = 1500
	}
	for _, opt := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		newFilter := func() *MatchedFilter {
			return NewMatchedFilter(opt, sub, MatchedFilterWindowSizeSubBlocks, numFilters,
				MatchedFilterAlignmentShiftSizeSubBlocks, 150, 0.7, 0.3, 0.2, true)
		}
		skip, ref := newFilter(), newFilter()
		ref.quietSkip = false
		s := newDlyQuietScene(41, 4, numFilters, math.Float32frombits(skip.quietLimitBits))
		// Lags 384 to 511 are in the windows of filter 0 and of filter 1.
		s.lag = MatchedFilterAlignmentShiftSizeSubBlocks*sub + 50
		quietWins := 0
		for blk := 0; blk < numBlocks; blk++ {
			mode := dlyQuietLoud
			if blk%120 >= 70 {
				mode = dlyQuietBelowLimit
			}
			y := s.block(mode, 150)
			updatesBefore := ref.numberPreEchoUpdates
			quiet := skip.detectPreEcho && skip.lastDetectedBestLagFilter >= 0 &&
				skip.filterIsQuiet(s.buf, skip.lastDetectedBestLagFilter)
			skip.Update(s.buf, y, false)
			ref.Update(s.buf, y, false)
			if d := dlyQuietStateDiff(skip, ref); d != "" {
				t.Fatalf("opt %d block %d: %s", opt, blk, d)
			}
			if quiet && ref.numberPreEchoUpdates > updatesBefore {
				quietWins++
			}
		}
		if quietWins < 20 {
			t.Fatalf("opt %d: the quiet accumulated-error filter won %d blocks; the scenario must make it win",
				opt, quietWins)
		}
	}
}

// TestDlyZeroFilterSkipStateMatchesNoSkip checks that a MatchedFilter with
// the skips and one without have the same state after every block of a
// capture that is ±0 for stretches, from the start and after resets, over
// loud, quiet and non-finite render. Some zero stretches start with a -0
// planted in a filter, which is not a zero filter, and some of their blocks
// have one sample that is not zero.
func TestDlyZeroFilterSkipStateMatchesNoSkip(t *testing.T) {
	const numFilters = 5
	numBlocks := 4000
	if testing.Short() {
		numBlocks = 1200
	}
	negZero := float32(math.Copysign(0, -1))
	for _, opt := range []Aec3Optimization{OptNone, OptNeon, OptSse2, OptAvx2} {
		for _, preEcho := range []bool{false, true} {
			sub := BlockSize / 4
			newFilter := func() *MatchedFilter {
				return NewMatchedFilter(opt, sub, MatchedFilterWindowSizeSubBlocks, numFilters,
					MatchedFilterAlignmentShiftSizeSubBlocks, 150, 0.7, 0.3, 0.2, preEcho)
			}
			skip, ref := newFilter(), newFilter()
			ref.quietSkip = false
			s := newDlyQuietScene(int64(60+int(opt)), 4, numFilters, math.Float32frombits(skip.quietLimitBits))
			zero, left, mode := true, 150, dlyQuietLoud
			for blk := 0; blk < numBlocks; blk++ {
				if left == 0 {
					zero = !zero
					left = 20 + s.rng.Intn(150)
					mode = []int{dlyQuietLoud, dlyQuietLoud, dlyQuietNaN, dlyQuietBelowLimit, dlyQuietSparseLoud}[s.rng.Intn(5)]
					if zero {
						// A zero stretch starts from zero filters, sometimes
						// with a -0 among the coefficients.
						skip.Reset(false)
						ref.Reset(false)
						if s.rng.Intn(3) == 0 {
							n, k := s.rng.Intn(numFilters), s.rng.Intn(skip.filterSize)
							skip.filter(n)[k] = negZero
							ref.filter(n)[k] = negZero
						}
					}
				}
				left--
				y := s.block(mode, 150)
				if zero {
					for i := range y {
						y[i] = 0
						if s.rng.Intn(4) == 0 {
							y[i] = negZero
						}
					}
					if s.rng.Intn(40) == 0 {
						// One sample before the last that is not zero: not a
						// zero block.
						y[s.rng.Intn(len(y)-1)] = []float32{0x1p-20, -0.5, 3}[s.rng.Intn(3)]
					}
				}
				slow := s.rng.Intn(3) == 0
				skip.Update(s.buf, y, slow)
				ref.Update(s.buf, y, slow)
				if d := dlyQuietStateDiff(skip, ref); d != "" {
					t.Fatalf("opt %d preEcho %v block %d (zero %v, mode %d): %s", opt, preEcho, blk, zero, mode, d)
				}
			}
			if n := skip.ZeroFilterSkipsForTesting(); n < uint64(numBlocks/8) {
				t.Errorf("opt %d preEcho %v: only %d zero-filter skips", opt, preEcho, n)
			}
			if ref.ZeroFilterSkipsForTesting() != 0 {
				t.Errorf("opt %d preEcho %v: the filter without the skips skipped", opt, preEcho)
			}
		}
	}
}
