package aec3

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The zero-gain skip of AdaptiveFirFilter and updateFrequencyResponse
// against a filter that always adapts and computes every response.

// afirSkipLanes lists the lanes this build can run.
func afirSkipLanes() []Aec3Optimization {
	lanes := []Aec3Optimization{OptNone}
	if isARM64 {
		lanes = append(lanes, OptNeon)
	}
	if isAMD64 {
		lanes = append(lanes, OptSse2)
		if DetectOptimization() == OptAvx2 {
			lanes = append(lanes, OptAvx2)
		}
	}
	return lanes
}

// afirSkipRig drives two filters with the same operations. skip takes the
// zero-gain skip; plain never does.
type afirSkipRig struct {
	opt         Aec3Optimization
	rng         *rand.Rand
	size, nch   int
	fb          *FftBuffer
	sb          *SpectrumBuffer
	rb          *RenderBuffer
	skip, plain *AdaptiveFirFilter
	h2Skip      [][FftLengthBy2Plus1]float32
	h2Plain     [][FftLengthBy2Plus1]float32
	irSkip      []float32
	irPlain     []float32
}

type afirSkipStats struct {
	steps         int
	skips         int // adaptations skipped
	negZeroAdapts int // skipped adaptations that still changed a coefficient
	nonFinite     int // zero gain with a non-finite render value in the window
	partialH2     int // frequency response updates of a part of the partitions
	negZeroCoeffs int // -0 coefficients seen in the filter
}

const (
	afirSkipBeforeAdapt = iota
	afirSkipAfterConstrain
)

func newAfirSkipRig(opt Aec3Optimization, seed int64) *afirSkipRig {
	rng := rand.New(rand.NewSource(seed))
	r := &afirSkipRig{opt: opt, rng: rng, size: 20, nch: 1 + rng.Intn(2)}
	r.fb = NewFftBuffer(r.size, r.nch)
	r.sb = NewSpectrumBuffer(r.size, r.nch)
	r.rb = NewRenderBuffer(NewBlockBuffer(r.size, 1, r.nch), r.sb, r.fb)
	// The spectra stay in step with the FFTs, as in a RenderDelayBuffer.
	r.rb.spectraMatchFfts = true
	const maxSize = 13
	initial := 6 + rng.Intn(maxSize-5)
	r.skip = NewAdaptiveFirFilter(maxSize, initial, 5, r.nch, opt)
	r.plain = NewAdaptiveFirFilter(maxSize, initial, 5, r.nch, opt)
	r.plain.skipDisabled = true
	r.h2Skip = make([][FftLengthBy2Plus1]float32, maxSize)
	r.h2Plain = make([][FftLengthBy2Plus1]float32, maxSize)
	r.irSkip = make([]float32, 0, GetTimeDomainLength(maxSize))
	r.irPlain = make([]float32, 0, GetTimeDomainLength(maxSize))
	return r
}

func afirSkipZero(rng *rand.Rand) float32 {
	return float32(math.Copysign(0, float64(rng.Intn(2))-0.5))
}

// insertRender moves the ring by one slot and writes a render FFT and its
// spectrum.
func (r *afirSkipRig) insertRender() {
	rng := r.rng
	pos := (r.fb.Read - 1 + r.size) % r.size
	mode := rng.Intn(20)
	for ch := range r.fb.At(pos) {
		x := &r.fb.At(pos)[ch]
		for k := range x.Re {
			switch {
			case mode < 4:
				x.Re[k], x.Im[k] = afirSkipZero(rng), afirSkipZero(rng)
			default:
				x.Re[k], x.Im[k] = float32(rng.NormFloat64()*100), float32(rng.NormFloat64()*100)
			}
		}
		if mode == 19 {
			v := []float32{float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}[rng.Intn(3)]
			if rng.Intn(2) == 0 {
				x.Re[rng.Intn(FftLengthBy2Plus1)] = v
			} else {
				x.Im[rng.Intn(FftLengthBy2Plus1)] = v
			}
		}
		x.Spectrum(r.opt, r.sb.At(pos)[ch][:])
	}
	r.fb.Read, r.sb.Read = pos, pos
}

func (r *afirSkipRig) windowFinite(n int) bool {
	for p := 0; p < n; p++ {
		for ch := range r.fb.At(0) {
			x := &r.fb.At((r.fb.Read + p) % r.size)[ch]
			for k := range x.Re {
				if v := x.Re[k] - x.Re[k] + x.Im[k] - x.Im[k]; v != 0 {
					return false
				}
			}
		}
	}
	return true
}

func afirSkipSameH(a, b FirCoefficients) (int, bool) {
	for i := range a.data {
		x, y := &a.data[i], &b.data[i]
		for k := range x.Re {
			if math.Float32bits(x.Re[k]) != math.Float32bits(y.Re[k]) ||
				math.Float32bits(x.Im[k]) != math.Float32bits(y.Im[k]) {
				return i, false
			}
		}
	}
	return 0, true
}

func (r *afirSkipRig) diff() string {
	a, b := r.skip, r.plain
	if a.currentSizePartitions != b.currentSizePartitions || a.partitionToConstrain != b.partitionToConstrain ||
		a.sizeChangeCounter != b.sizeChangeCounter {
		return "sizes or constrained partition differ"
	}
	if i, ok := afirSkipSameH(a.H, b.H); !ok {
		return fmt.Sprintf("H differs in element %d", i)
	}
	if len(r.h2Skip) != len(r.h2Plain) {
		return "H2 lengths differ"
	}
	for p := range r.h2Skip {
		for k := range r.h2Skip[p] {
			if math.Float32bits(r.h2Skip[p][k]) != math.Float32bits(r.h2Plain[p][k]) {
				return fmt.Sprintf("H2 differs in partition %d, bin %d", p, k)
			}
		}
	}
	if len(r.irSkip) != len(r.irPlain) {
		return "impulse response lengths differ"
	}
	for i := range r.irSkip {
		if math.Float32bits(r.irSkip[i]) != math.Float32bits(r.irPlain[i]) {
			return fmt.Sprintf("impulse response differs at %d", i)
		}
	}
	return ""
}

// step runs one block on both filters and returns the first difference.
func (r *afirSkipRig) step(st *afirSkipStats, tamper func(stage int, r *afirSkipRig)) string {
	rng := r.rng
	st.steps++
	r.insertRender()
	if tamper != nil {
		tamper(afirSkipBeforeAdapt, r)
	}

	// The caller of the filter sums the spectra of the current size.
	n := r.skip.SizePartitions()
	var X2 [FftLengthBy2Plus1]float32
	r.rb.SpectralSum(n, &X2)

	var G FftData
	zeroGain := true
	switch g := rng.Intn(10); {
	case g < 6:
		for k := range G.Re {
			G.Re[k], G.Im[k] = afirSkipZero(rng), afirSkipZero(rng)
		}
	case g < 7:
		// Zero but for one tiny element.
		zeroGain = false
		for k := range G.Re {
			G.Re[k], G.Im[k] = afirSkipZero(rng), afirSkipZero(rng)
		}
		if rng.Intn(2) == 0 {
			G.Re[rng.Intn(FftLengthBy2Plus1)] = math.SmallestNonzeroFloat32
		} else {
			G.Im[rng.Intn(FftLengthBy2Plus1)] = -math.SmallestNonzeroFloat32
		}
	default:
		zeroGain = false
		afirTestRandFft(rng, &G, 1e-4)
	}

	before := r.skip.H.Clone()
	skipsBefore := r.skip.zeroGainSkips
	r.skip.adaptAndUpdateSizeKnownRender(r.rb, &G, &X2, n)
	r.plain.adaptAndUpdateSize(r.rb, &G)
	if r.skip.zeroGainSkips != skipsBefore {
		st.skips++
		if _, same := afirSkipSameH(before, r.skip.H); !same {
			st.negZeroAdapts++
		}
	} else if zeroGain && !r.windowFinite(r.skip.SizePartitions()) {
		st.nonFinite++
	}
	if d := r.diff(); d != "" {
		return "after the adaptation: " + d
	}

	switch op := rng.Intn(100); {
	case op < 40:
		r.skip.constrain()
		r.plain.constrain()
	case op < 75:
		r.skip.constrainAndUpdateImpulseResponse(&r.irSkip)
		r.plain.constrainAndUpdateImpulseResponse(&r.irPlain)
	case op < 80:
		factor := []float32{0.5, -1, 0, afirSkipZero(rng), 1e-30, -1e-30}[rng.Intn(6)]
		r.skip.ScaleFilter(factor)
		r.plain.ScaleFilter(factor)
	case op < 86:
		H := afirTestH(rng, r.skip.maxSizePartitions, r.nch)
		for i := range H.data {
			for k := range H.data[i].Re {
				if rng.Intn(3) == 0 {
					H.data[i].Re[k] = afirSkipZero(rng)
				}
				if rng.Intn(3) == 0 {
					H.data[i].Im[k] = afirSkipZero(rng)
				}
			}
		}
		np := rng.Intn(r.skip.maxSizePartitions + 1)
		r.skip.SetFilter(np, H)
		r.plain.SetFilter(np, H)
	case op < 91:
		size, immediate := 1+rng.Intn(r.skip.maxSizePartitions+2), rng.Intn(2) == 0
		r.skip.SetSizePartitions(size, immediate)
		r.plain.SetSizePartitions(size, immediate)
	case op < 94:
		r.skip.HandleEchoPathChange()
		r.plain.HandleEchoPathChange()
	}
	if tamper != nil {
		tamper(afirSkipAfterConstrain, r)
	}

	if !r.skip.h2Stale.allBelow(r.skip.SizePartitions()) {
		st.partialH2++
	}
	r.skip.updateFrequencyResponse(&r.h2Skip)
	r.plain.ComputeFrequencyResponse(&r.h2Plain)
	if afirHoldsNegZero(r.plain.H.data) {
		st.negZeroCoeffs++
	}
	return r.diff()
}

func afirSkipRun(opt Aec3Optimization, seed int64, steps int, tamper func(stage int, r *afirSkipRig)) (string, afirSkipStats) {
	r := newAfirSkipRig(opt, seed)
	var st afirSkipStats
	for i := 0; i < steps; i++ {
		if d := r.step(&st, tamper); d != "" {
			return fmt.Sprintf("step %d: %s", i, d), st
		}
	}
	return "", st
}

// TestAfirZeroGainSkipStateMatchesNoSkip checks that the coefficients, the
// frequency responses and the impulse responses of the two filters agree bit
// for bit after every operation, in every lane. The runs take the skip with
// and without -0 coefficients, meet render values that are not finite under
// a zero gain, and update the frequency response in part.
func TestAfirZeroGainSkipStateMatchesNoSkip(t *testing.T) {
	steps := 3000
	if testing.Short() {
		steps = 600
	}
	for _, opt := range afirSkipLanes() {
		var total afirSkipStats
		for seed := int64(1); seed <= 12; seed++ {
			d, st := afirSkipRun(opt, seed, steps, nil)
			if d != "" {
				t.Fatalf("lane %v, seed %d: %s", opt, seed, d)
			}
			total.steps += st.steps
			total.skips += st.skips
			total.negZeroAdapts += st.negZeroAdapts
			total.nonFinite += st.nonFinite
			total.partialH2 += st.partialH2
			total.negZeroCoeffs += st.negZeroCoeffs
		}
		t.Logf("lane %v: %+v", opt, total)
		if total.skips < total.steps/10 || total.negZeroAdapts < 20 || total.nonFinite < total.steps/50 ||
			total.partialH2 < total.steps/10 || total.negZeroCoeffs < total.steps/20 ||
			total.skips-total.negZeroAdapts < total.steps/20 {
			t.Fatalf("lane %v: the runs do not cover the skip: %+v", opt, total)
		}
	}
}

// TestAfirZeroGainSkipMutationsAreDetected checks the test above: a skip
// that ignores one of its conditions, or a frequency response update that
// misses a write, makes the two filters differ.
func TestAfirZeroGainSkipMutationsAreDetected(t *testing.T) {
	mutations := []struct {
		name   string
		tamper func(stage int, r *afirSkipRig)
	}{
		{"-0 coefficients are not tracked", func(stage int, r *afirSkipRig) {
			if stage == afirSkipBeforeAdapt {
				r.skip.negZero = afirPartitionSet{}
			}
		}},
		{"the spectra do not show a non-finite render value", func(stage int, r *afirSkipRig) {
			if stage == afirSkipBeforeAdapt {
				clear(r.sb.data)
			}
		}},
		{"the response of a written partition is kept", func(stage int, r *afirSkipRig) {
			if stage == afirSkipAfterConstrain {
				r.skip.h2Stale = afirPartitionSet{}
			}
		}},
	}
	for _, opt := range afirSkipLanes() {
		for _, m := range mutations {
			detected := 0
			for seed := int64(1); seed <= 6; seed++ {
				if d, _ := afirSkipRun(opt, seed, 600, m.tamper); d != "" {
					detected++
				}
			}
			if detected < 6 {
				t.Fatalf("lane %v: mutation %q went undetected in %d of 6 runs", opt, m.name, 6-detected)
			}
		}
	}
}

// TestAfirZeroGainSkipNeedsMatchingSpectra checks that the skip is not taken
// with a render buffer whose spectra are not known to match its FFTs, with a
// sum over fewer partitions than the filter has, and with a filter beyond
// the tracked size.
func TestAfirZeroGainSkipNeedsMatchingSpectra(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	rb := afirTestRenderBuffer(rng, 20, 1, 3)
	var G FftData
	var X2 [FftLengthBy2Plus1]float32
	f := NewAdaptiveFirFilter(12, 12, 5, 1, OptNone)
	f.adaptAndUpdateSizeKnownRender(rb, &G, &X2, 12)
	if f.zeroGainSkips != 0 {
		t.Fatal("skipped on a render buffer whose spectra need not match")
	}
	rb.spectraMatchFfts = true
	rb.spectrumBuffer.Read = rb.fftBuffer.Read
	f.adaptAndUpdateSizeKnownRender(rb, &G, &X2, 11)
	f.adaptAndUpdateSizeKnownRender(rb, &G, nil, 12)
	if f.zeroGainSkips != 0 {
		t.Fatal("skipped without a sum over all partitions")
	}
	rb.spectrumBuffer.Read = (rb.fftBuffer.Read + 1) % 20
	f.adaptAndUpdateSizeKnownRender(rb, &G, &X2, 12)
	if f.zeroGainSkips != 0 {
		t.Fatal("skipped with rings read at different positions")
	}
	rb.spectrumBuffer.Read = rb.fftBuffer.Read
	f.adaptAndUpdateSizeKnownRender(rb, &G, &X2, 12)
	if f.zeroGainSkips != 1 {
		t.Fatal("did not skip")
	}
	big := NewAdaptiveFirFilter(afirTrackedPartitions+1, 4, 5, 1, OptNone)
	bigRb := afirTestRenderBuffer(rng, 300, 1, 0)
	bigRb.spectraMatchFfts = true
	big.adaptAndUpdateSizeKnownRender(bigRb, &G, &X2, afirTrackedPartitions+1)
	if big.zeroGainSkips != 0 {
		t.Fatal("skipped in a filter beyond the tracked size")
	}
}

func TestAfirPartitionSet(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	for trial := 0; trial < 2000; trial++ {
		var s afirPartitionSet
		var ref [afirTrackedPartitions]bool
		for i := 0; i < 40; i++ {
			p, n := rng.Intn(afirTrackedPartitions), rng.Intn(afirTrackedPartitions+1)
			switch rng.Intn(4) {
			case 0:
				s.add(p)
				ref[p] = true
			case 1:
				s.remove(p)
				ref[p] = false
			case 2:
				s.addBelow(n)
				for q := 0; q < n; q++ {
					ref[q] = true
				}
			case 3:
				s.removeBelow(n)
				for q := 0; q < n; q++ {
					ref[q] = false
				}
			}
			m := rng.Intn(afirTrackedPartitions + 1)
			anyRef, allRef := false, true
			for q := 0; q < m; q++ {
				anyRef = anyRef || ref[q]
				allRef = allRef && ref[q]
			}
			if s.anyBelow(m) != anyRef || s.allBelow(m) != allRef || s.has(p) != ref[p] {
				t.Fatalf("trial %d: set disagrees with the reference below %d", trial, m)
			}
		}
	}
}

// afirSkipProcessors builds two block processors in lane opt that differ
// only in the zero-gain skip.
func afirSkipProcessors(cfg *EchoCanceller3Config, opt Aec3Optimization, rate, nr, nc int) (skip, plain *BlockProcessor, skipEr, plainEr *EchoRemover) {
	build := func(disabled bool) (*BlockProcessor, *EchoRemover) {
		er := NewEchoRemoverWithOptimization(cfg, rate, nr, nc, opt)
		channels := er.subtractor.channelStates()
		for ch := range channels {
			channels[ch].refinedFilter.skipDisabled = disabled
			channels[ch].coarseFilter.skipDisabled = disabled
		}
		return NewBlockProcessorWithComponents(cfg, rate, nr, nc, NewRenderDelayBuffer(cfg, rate, nr),
			NewRenderDelayController(cfg, rate, nc), er), er
	}
	skip, skipEr = build(false)
	plain, plainEr = build(true)
	return
}

// TestEc3ZeroGainSkipStateMatchesNoSkip runs two block processors that
// differ only in the skip and checks that the output and the subtractor
// state agree after every block. The signal has speech-like noise, digital
// silence, quiet noise, and a silent and a saturated capture.
func TestEc3ZeroGainSkipStateMatchesNoSkip(t *testing.T) {
	type segment struct {
		blocks                   int
		render, capture, nearend float64 // amplitudes
		saturated                bool
		nan                      bool
	}
	schedule := []segment{
		{blocks: 300, render: 3000, capture: 0.5, nearend: 30},
		{blocks: 150, capture: 0.5, nearend: 30},
		{blocks: 150, render: 8, capture: 0.5, nearend: 30},
		{blocks: 200, render: 3000, capture: 0.5, nearend: 30},
		{blocks: 150, render: 3000},
		{blocks: 100},
		{blocks: 100, render: 3000, capture: 0.5, nearend: 30, saturated: true},
		{blocks: 200, render: 3000, capture: 0.5, nearend: 30},
		{blocks: 60, capture: 0.5, nearend: 30},
		{blocks: 3, render: 3000, capture: 0.5, nearend: 30, nan: true},
		{blocks: 60, capture: 0.5, nearend: 30},
	}
	if testing.Short() {
		for i := range schedule {
			schedule[i].blocks = (schedule[i].blocks + 2) / 3
		}
	}
	for _, opt := range []Aec3Optimization{OptNone, DetectOptimization()} {
		for _, tc := range []struct{ rate, nr, nc int }{{16000, 1, 1}, {48000, 2, 2}} {
			cfg := DefaultEchoCanceller3Config()
			skip, plain, skipEr, plainEr := afirSkipProcessors(&cfg, opt, tc.rate, tc.nr, tc.nc)
			nb := NumBandsForRate(tc.rate)
			rng := rand.New(rand.NewSource(9))
			const delay = 5
			history := make([][]float32, delay+1) // band 0 of render channel 0
			for i := range history {
				history[i] = make([]float32, BlockSize)
			}
			block, loudSkips, quietBlocks := 0, uint64(0), 0
			for si, seg := range schedule {
				skipsBefore := skipEr.subtractor.ZeroGainSkipsForTesting()
				for i := 0; i < seg.blocks; i, block = i+1, block+1 {
					render := NewBlock(nb, tc.nr)
					for b := 0; b < nb; b++ {
						for ch := 0; ch < tc.nr; ch++ {
							v := render.View(b, ch)
							for k := range v {
								v[k] = float32(seg.render * rng.NormFloat64())
							}
						}
					}
					if seg.nan {
						render.View(0, 0)[rng.Intn(BlockSize)] = float32(math.NaN())
					}
					copy(history[block%len(history)], render.View(0, 0))
					echo := history[(block+1)%len(history)]
					captureSkip, capturePlain := NewBlock(nb, tc.nc), NewBlock(nb, tc.nc)
					for b := 0; b < nb; b++ {
						for ch := 0; ch < tc.nc; ch++ {
							v := captureSkip.View(b, ch)
							for k := range v {
								v[k] = float32(seg.nearend * rng.NormFloat64())
								if b == 0 {
									v[k] += float32(seg.capture) * echo[k]
								}
							}
							copy(capturePlain.View(b, ch), v)
						}
					}
					renderPlain := NewBlock(nb, tc.nr)
					for b := 0; b < nb; b++ {
						for ch := 0; ch < tc.nr; ch++ {
							copy(renderPlain.View(b, ch), render.View(b, ch))
						}
					}
					skip.BufferRender(&render)
					plain.BufferRender(&renderPlain)
					skip.ProcessCapture(false, seg.saturated, nil, &captureSkip)
					plain.ProcessCapture(false, seg.saturated, nil, &capturePlain)
					for b := 0; b < nb; b++ {
						for ch := 0; ch < tc.nc; ch++ {
							afirEqualArr(t, fmt.Sprintf("lane %v, %d Hz, segment %d, block %d: output", opt, tc.rate, si, i),
								captureSkip.View(b, ch), capturePlain.View(b, ch))
						}
					}
					a, p := skipEr.subtractor.channelStates(), plainEr.subtractor.channelStates()
					for ch := range a {
						where := fmt.Sprintf("lane %v, %d Hz, segment %d, block %d, channel %d", opt, tc.rate, si, i, ch)
						if _, ok := afirSkipSameH(a[ch].refinedFilter.H, p[ch].refinedFilter.H); !ok {
							t.Fatalf("%s: refined filters differ", where)
						}
						if _, ok := afirSkipSameH(a[ch].coarseFilter.H, p[ch].coarseFilter.H); !ok {
							t.Fatalf("%s: coarse filters differ", where)
						}
						h2a, h2p := skipEr.subtractor.FilterFrequencyResponses()[ch], plainEr.subtractor.FilterFrequencyResponses()[ch]
						if len(h2a) != len(h2p) {
							t.Fatalf("%s: frequency response sizes differ", where)
						}
						for q := range h2a {
							afirEqualArr(t, where+": frequency response", h2a[q][:], h2p[q][:])
						}
						afirEqualArr(t, where+": impulse response", skipEr.subtractor.FilterImpulseResponses()[ch],
							plainEr.subtractor.FilterImpulseResponses()[ch])
					}
				}
				n := skipEr.subtractor.ZeroGainSkipsForTesting() - skipsBefore
				if seg.render >= 3000 && seg.capture > 0 && !seg.saturated && !seg.nan && si > 0 {
					loudSkips += n
				}
				if seg.render < 3000 {
					quietBlocks += seg.blocks
				}
				t.Logf("lane %v, %d Hz, segment %d (%+v): %d of %d adaptations skipped", opt, tc.rate, si, seg, n,
					2*tc.nc*seg.blocks)
			}
			skips := skipEr.subtractor.ZeroGainSkipsForTesting()
			if plainEr.subtractor.ZeroGainSkipsForTesting() != 0 {
				t.Fatalf("lane %v: the reference skipped", opt)
			}
			if skips < uint64(quietBlocks) || skips >= uint64(2*tc.nc*block) {
				t.Fatalf("lane %v, %d Hz: %d of %d adaptations skipped; the schedule must cover both paths", opt, tc.rate,
					skips, 2*tc.nc*block)
			}
			_ = loudSkips
		}
	}
}
