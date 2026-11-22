package aec3

import (
	"math"
	"sync/atomic"
)

// dlyAccumulatedErrorSubSampleRate is the subsampling of the accumulated
// error. The cores depend on its value.
const dlyAccumulatedErrorSubSampleRate = 4

func dlyUpdateAccumulatedError(instantaneousAccumulatedError, accumulatedError []float32, oneOverErrorSumAnchor float32) {
	const smoothConstantIncreases = 0.015
	accumulatedError = accumulatedError[:len(instantaneousAccumulatedError)]
	for k, v := range instantaneousAccumulatedError {
		errorNorm := float32(v * oneOverErrorSumAnchor)
		if errorNorm < accumulatedError[k] {
			accumulatedError[k] = errorNorm
		} else {
			accumulatedError[k] += float32(smoothConstantIncreases * (errorNorm - accumulatedError[k]))
		}
	}
}

func dlyComputePreEchoLag(accumulatedError []float32, lag, alignmentShiftWinner int) int {
	const preEchoThreshold = 0.5
	preEchoLagEstimate := lag - alignmentShiftWinner
	maximumPreEchoLag := min(preEchoLagEstimate/dlyAccumulatedErrorSubSampleRate, len(accumulatedError))
	for k := maximumPreEchoLag - 1; k >= 0; k-- {
		if accumulatedError[k] > preEchoThreshold {
			break
		}
		preEchoLagEstimate = (k+1)*dlyAccumulatedErrorSubSampleRate - 1
	}
	return preEchoLagEstimate + alignmentShiftWinner
}

func dlySaturated(y float32) bool { return y >= 32000 || y <= -32000 }

// MatchedFilterCoreWithAccumulatedErrorNeon mirrors
// aec3::MatchedFilterCoreWithAccumulatedError_NEON. scratchMemory holds
// len(h)+len(y) samples.
func MatchedFilterCoreWithAccumulatedErrorNeon(xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, accumulatedError, scratchMemory []float32) {
	dlyAccErrCore(4, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
		accumulatedError, scratchMemory)
}

// MatchedFilterCoreNeon mirrors aec3::MatchedFilterCore_NEON.
func MatchedFilterCoreNeon(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		dlyAccErrCore(4, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			accumulatedError, scratchMemory)
		return
	}
	dlyCoreLane(4, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
}

// MatchedFilterCoreSse2 mirrors aec3::MatchedFilterCore_SSE2.
func MatchedFilterCoreSse2(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		dlyAccErrCore(8, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			accumulatedError, scratchMemory)
		return
	}
	dlyCoreLane(8, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
}

// MatchedFilterCoreAvx2 mirrors aec3::MatchedFilterCore_AVX2.
func MatchedFilterCoreAvx2(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		dlyAccErrCore(16, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			accumulatedError, scratchMemory)
		return
	}
	dlyCoreLane(16, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum)
}

// The lane cores run the sample loop of MatchedFilterCore_NEON (group size g
// = 4), _SSE2 (8) and _AVX2 (16). Per capture sample the C++ accumulates h*x
// and x*x over the whole groups of each of the two chunks of the circular
// buffer in vector accumulators and over the rest in scalars, reduces, and
// updates the filter in a second pass. The Go cores compute the same values
// in another arrangement.
//
// x*x and h*x are separate accumulations (matched_filter_kernels.go).
//
// While the window of a filter does not wrap and its length is a multiple of
// g, lane l of the x*x accumulators of the sample with start index a is the
// chain x[a+l]^2, x[a+l+g]^2, ... from +0. It depends on a+l only, so
// consecutive samples share all but one lane: dlyX2Sums computes each chain
// once for a run of samples and reduces the g lanes of every sample in the
// order of the lane. dlyX2Wrapped computes the sums of the other samples.
//
// The update of a sample is deferred into the h*x accumulation of the next
// one, or a final pass, which updates each element of h right before reading
// it. An element sees the operations of a separate update pass.
//
// Two filters run their samples in step (dlyCoreLanes), so that a kernel can
// interleave their accumulations. Filters share no state.

// dlyDotS accumulates h*x into the s accumulators of the lane with group
// size g, which are the front of acc.
func dlyDotS(g int, h, x []float32, acc *[16]float32) {
	switch g {
	case 4:
		dlyDotSNeon(h, x, (*[4]float32)(acc[:4]))
	case 8:
		dlyDotSSse2(h, x, (*[8]float32)(acc[:8]))
	default:
		dlyDotSAvx2(h, x, acc)
	}
}

// dlyAxpyDotS applies the filter update of the lane, h += alpha*xa, and
// accumulates the updated h against xb. The AVX2 lane fuses the update.
func dlyAxpyDotS(g int, h, xa, xb []float32, alpha float32, acc *[16]float32) {
	switch g {
	case 4:
		dlyAxpyDotSNeon(h, xa, xb, alpha, (*[4]float32)(acc[:4]))
	case 8:
		dlyAxpyDotSSse2(h, xa, xb, alpha, (*[8]float32)(acc[:8]))
	default:
		dlyAxpyFmaDotSAvx2(h, xa, xb, alpha, acc)
	}
}

// dlySq accumulates x*x into the accumulators of the lane.
func dlySq(g int, x []float32, acc *[16]float32) {
	switch g {
	case 4:
		dlySqNeon(x, (*[4]float32)(acc[:4]))
	case 8:
		dlySqSse2(x, (*[8]float32)(acc[:8]))
	default:
		dlySqAvx2(x, acc)
	}
}

// dlyReduceS adds the lane accumulators of one sum to tail, the sum of the
// scalar tails, in the order of the C++ reduction of the lane: pairwise for
// NEON, ((v0+v1)+v2)+v3 on the sum of the two accumulators for SSE2, hsum_ab
// on that sum for AVX2.
func dlyReduceS(g int, acc *[16]float32, tail float32) float32 {
	switch g {
	case 4:
		return tail + ((acc[0] + acc[1]) + (acc[2] + acc[3]))
	case 8:
		var v [4]float32
		for l := range v {
			v[l] = acc[l] + acc[4+l]
		}
		return tail + (((v[0] + v[1]) + v[2]) + v[3])
	}
	var b [8]float32
	for l := range b {
		b[l] = acc[l] + acc[8+l]
	}
	return tail + (((b[0] + b[1]) + (b[2] + b[3])) + ((b[4] + b[5]) + (b[6] + b[7])))
}

// dlyReduceX2 reduces the x*x accumulators. The accumulated-error core of
// the AVX2 lane adds the two halves of the sum of the accumulators and then
// the four lanes in sequence; every other core reduces x*x like h*x.
func dlyReduceX2(g int, accErr bool, acc *[16]float32, tail float32) float32 {
	if !accErr || g != 16 {
		return dlyReduceS(g, acc, tail)
	}
	var a [8]float32
	for l := range a {
		a[l] = acc[l] + acc[8+l]
	}
	var v [4]float32
	for l := range v {
		v[l] = a[l] + a[4+l]
	}
	return tail + (((v[0] + v[1]) + v[2]) + v[3])
}

// dlyX2Run is the number of capture samples one table of x*x sums covers.
const dlyX2Run = 16

// dlyX2LanesRef stores in lanes[a] the x*x lane chain that starts at x[a]:
// x[a]^2, x[a+g]^2, ..., n/g terms accumulated from +0. len(lanes) >= g and
// len(x) >= len(lanes)+n-g.
func dlyX2LanesRef(g int, x []float32, n int, lanes []float32) {
	last := len(lanes) - g
	for o := 0; ; o += g {
		o = min(o, last)
		var acc [16]float32
		dlySq(g, x[o:o+n], &acc)
		copy(lanes[o:o+g], acc[:g])
		if o == last {
			return
		}
	}
}

// dlyX2Sums stores in out[j], for j < r, the x*x sum the lane with group
// size g computes for the window x[j:j+n]. n is a multiple of g and len(x)
// >= n+r-1.
func dlyX2Sums(g int, accErr bool, x []float32, n, r int, out *[dlyX2Run]float32) {
	var lanes [dlyX2Run + 15]float32
	dlyX2Lanes(g, x, n, lanes[:r+g-1])
	dlyX2Reduce(g, accErr, &lanes, r, out)
}

// The x*x sums of the unfused lanes from a table of squares.
//
// A sample of a stretch belongs to the chains of up to g windows (dlyX2Sums)
// or to the accumulators of up to dlyX2Run windows (dlyX2Wrapped). The Sq
// routines square every sample once, into scratch, and then add, eight
// chains side by side. The squares are rounded to float32 before they are
// added, as in the unfused lanes; the AVX2 lane fuses the two and takes no
// part.

// dlyX2SqScratch returns the scratch the Sq routines need for filters of
// hSize coefficients.
func dlyX2SqScratch(hSize int) int { return hSize + dlyX2Run + 8 }

// dlyX2Chains8 stores in out[j], for j < 8, the sum of sq[j], sq[j+g],
// sq[j+2g], ..., n/g terms accumulated in order from +0, for g = 4 or 8.
// len(sq) >= n-g+8.
func dlyX2Chains8(sq []float32, g, n int, out *[8]float32) {
	var q0, q1, q2, q3, q4, q5, q6, q7 float32
	if n >= g {
		sq = sq[: n-g+8 : n-g+8]
		if g == 4 {
			for k := 0; k < len(sq)-7; k += 4 {
				w := (*[8]float32)(sq[k : k+8])
				q0 += w[0]
				q1 += w[1]
				q2 += w[2]
				q3 += w[3]
				q4 += w[4]
				q5 += w[5]
				q6 += w[6]
				q7 += w[7]
			}
		} else {
			for k := 0; k < len(sq)-7; k += 8 {
				w := (*[8]float32)(sq[k : k+8])
				q0 += w[0]
				q1 += w[1]
				q2 += w[2]
				q3 += w[3]
				q4 += w[4]
				q5 += w[5]
				q6 += w[6]
				q7 += w[7]
			}
		}
	}
	*out = [8]float32{q0, q1, q2, q3, q4, q5, q6, q7}
}

// dlyX2SumsSq is dlyX2Sums, without accErr, for g = 4 or 8, with the squares
// of x[:n+r-1] in scratch. len(scratch) >= dlyX2SqScratch(n).
func dlyX2SumsSq(g int, x []float32, n, r int, scratch []float32, out *[dlyX2Run]float32) {
	span := n + r - 1
	sq := scratch[:dlyX2SqScratch(n)]
	for i, v := range x[:span] {
		sq[i] = float32(v * v)
	}
	// The chains run eight at a time. Those beyond the r+g-1 the windows use
	// read scratch past the squares, and their sums are not used.
	var lanes [dlyX2Run + 15]float32
	for o := 0; o < r+g-1; o += 8 {
		dlyX2Chains8(sq[o:], g, n, (*[8]float32)(lanes[o:o+8]))
	}
	dlyX2Reduce(g, false, &lanes, r, out)
}

// dlyAddLanes adds s[k+l] to acc[l] for the whole groups of g = 4 or 8
// values of s, in order.
func dlyAddLanes(g int, s []float32, acc *[16]float32) {
	if g == 4 {
		n := len(s) &^ 3
		s = s[:n:n]
		q0, q1, q2, q3 := acc[0], acc[1], acc[2], acc[3]
		for k := 0; k < len(s)-3; k += 4 {
			w := (*[4]float32)(s[k : k+4])
			q0 += w[0]
			q1 += w[1]
			q2 += w[2]
			q3 += w[3]
		}
		acc[0], acc[1], acc[2], acc[3] = q0, q1, q2, q3
		return
	}
	n := len(s) &^ 7
	s = s[:n:n]
	q0, q1, q2, q3, q4, q5, q6, q7 := acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]
	for k := 0; k < len(s)-7; k += 8 {
		w := (*[8]float32)(s[k : k+8])
		q0 += w[0]
		q1 += w[1]
		q2 += w[2]
		q3 += w[3]
		q4 += w[4]
		q5 += w[5]
		q6 += w[6]
		q7 += w[7]
	}
	acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7] = q0, q1, q2, q3, q4, q5, q6, q7
}

// dlyX2WrappedSq is dlyX2WrappedRef for g = 4 or 8, with the squares of the
// hSize+r-1 samples from lo on, around the end of the buffer, in scratch:
// the window of start index lo+t is scratch[t:t+hSize]. hSize+r-1 <= len(x)
// and len(scratch) >= dlyX2SqScratch(hSize).
func dlyX2WrappedSq(g int, x []float32, hSize, lo, r int, scratch []float32, out *[dlyX2Run]float32) {
	span := hSize + r - 1
	sq := scratch[:span]
	n1 := min(span, len(x)-lo)
	for i, v := range x[lo : lo+n1] {
		sq[i] = float32(v * v)
	}
	for i, v := range x[:span-n1] {
		sq[n1+i] = float32(v * v)
	}
	for t := 0; t < r; t++ {
		chunk1 := min(hSize, len(x)-(lo+t))
		v1 := chunk1 &^ (g - 1)
		v2 := (hSize - chunk1) &^ (g - 1)
		w := sq[t : t+hSize]
		var q [16]float32
		dlyAddLanes(g, w[:v1], &q)
		dlyAddLanes(g, w[chunk1:chunk1+v2], &q)
		var tail float32
		for _, v := range w[v1:chunk1] {
			tail += v
		}
		for _, v := range w[chunk1+v2:] {
			tail += v
		}
		out[t] = dlyReduceX2(g, false, &q, tail)
	}
}

// dlyX2SumsWide is dlyX2Sums of the AVX2 lane, without accErr, on the
// widened samples xw.
func dlyX2SumsWide(xw []float64, n, r int, out *[dlyX2Run]float32) {
	var lanes [dlyX2Run + 15]float32
	last := r - 1
	for o := 0; ; o += 16 {
		o = min(o, last)
		var acc [16]float32
		dlySqAvx2Wide(xw[o:o+n], &acc)
		copy(lanes[o:o+16], acc[:])
		if o == last {
			break
		}
	}
	dlyX2Reduce(16, false, &lanes, r, out)
}

// dlyX2WrappedWide is dlyX2WrappedRef of the AVX2 lane with the whole groups
// read from the widened samples xw.
func dlyX2WrappedWide(x []float32, xw []float64, hSize, lo, r int, out *[dlyX2Run]float32) {
	for t := 0; t < r; t++ {
		a := lo + t
		chunk1 := min(hSize, len(x)-a)
		v1 := chunk1 &^ 15
		v2 := (hSize - chunk1) &^ 15
		var q [16]float32
		dlySqAvx2Wide(xw[a:a+v1], &q)
		dlySqAvx2Wide(xw[:v2], &q)
		out[t] = dlyReduceX2(16, false, &q, dlyX2Tails(16, x, hSize, a))
	}
}

// dlyX2Lanes2Ref is dlyX2Lanes for two stretches of the same length.
func dlyX2Lanes2Ref(g int, xA, xB []float32, n int, lanesA, lanesB []float32) {
	dlyX2Lanes(g, xA, n, lanesA)
	dlyX2Lanes(g, xB, n, lanesB)
}

// x2TableRun returns the run of samples the next table of dlyX2Sums covers:
// the start index lo of the last of them and their number r. ok reports
// whether the current sample needs such a table: its window does not wrap,
// the filter length is a multiple of g, and no table holds its sum.
func (f *dlyLaneFilter) x2TableRun(g int, x []float32, left int) (lo, r int, ok bool) {
	hSize, xs := len(f.h), f.xStart
	if xs+hSize > len(x) || hSize&(g-1) != 0 || (xs >= f.x2Lo && xs < f.x2Hi) {
		return 0, 0, false
	}
	// The samples that follow have the start indices below xs, down to zero.
	// None of their windows wraps.
	r = min(left, xs+1, dlyX2Run)
	return xs + 1 - r, r, true
}

// dlyX2Tables2 fills the tables of x*x sums of two filters at once when both
// need one for runs of one length, so that a kernel can interleave their
// lane chains.
func dlyX2Tables2(g int, fa, fb *dlyLaneFilter, x []float32, left int) {
	loA, rA, okA := fa.x2TableRun(g, x, left)
	loB, rB, okB := fb.x2TableRun(g, x, left)
	if !okA || !okB || rA != rB || len(fa.h) != len(fb.h) {
		return
	}
	var lanesA, lanesB [dlyX2Run + 15]float32
	dlyX2Lanes2(g, x[loA:], x[loB:], len(fa.h), lanesA[:rA+g-1], lanesB[:rA+g-1])
	dlyX2Reduce(g, false, &lanesA, rA, &fa.x2)
	dlyX2Reduce(g, false, &lanesB, rA, &fb.x2)
	fa.x2Lo, fa.x2Hi = loA, loA+rA
	fb.x2Lo, fb.x2Hi = loB, loB+rA
}

// dlyX2ReduceRef stores in out[j], for j < r, the reduction of the g lanes
// lanes[j:j+g]: dlyReduceX2 with a zero tail.
func dlyX2ReduceRef(g int, accErr bool, lanes *[dlyX2Run + 15]float32, r int, out *[dlyX2Run]float32) {
	if g == 4 {
		// dlyReduceS without the copy.
		for j := 0; j < r; j++ {
			l := (*[4]float32)(lanes[j : j+4])
			out[j] = 0 + ((l[0] + l[1]) + (l[2] + l[3]))
		}
		return
	}
	for j := 0; j < r; j++ {
		var a [16]float32
		copy(a[:g], lanes[j:j+g])
		out[j] = dlyReduceX2(g, accErr, &a, 0)
	}
}

// dlyX2WrappedRef stores in out[a-lo], for the r start indices a from lo,
// the x*x sum the lane with group size g computes for the window of hSize
// samples at a in the circular buffer x, as the C++ loop lays it out: the
// whole groups of the chunk up to the end of the buffer and then those of
// the chunk from its start in the lane accumulators, the rest of each chunk
// in a scalar. It serves the windows that wrap and the filter lengths that
// are not a multiple of g.
func dlyX2WrappedRef(g int, x []float32, hSize, lo, r int, out *[dlyX2Run]float32) {
	for t := 0; t < r; t++ {
		a := lo + t
		chunk1 := min(hSize, len(x)-a)
		v1 := chunk1 &^ (g - 1)
		v2 := (hSize - chunk1) &^ (g - 1)
		var q [16]float32
		dlySq(g, x[a:a+v1], &q)
		dlySq(g, x[:v2], &q)
		out[t] = dlyReduceX2(g, false, &q, dlyX2Tails(g, x, hSize, a))
	}
}

// dlyMaxStreams is the largest number of stretches dlySqMulti accumulates at
// once.
const dlyMaxStreams = 5

// dlySqMultiRef runs dlySq on the k stretches x[offs[t]:offs[t]+n] with the
// accumulators acc[t].
func dlySqMultiRef(g int, x []float32, n int, offs *[dlyMaxStreams]int, acc *[dlyMaxStreams][16]float32, k int) {
	for t := 0; t < k; t++ {
		dlySq(g, x[offs[t]:offs[t]+n], &acc[t])
	}
}

// dlyX2LanesMulti is dlyX2LanesRef with the chains of dlySqStreams(g)
// groups of g lanes accumulated in one loop.
func dlyX2LanesMulti(g int, x []float32, n int, lanes []float32) {
	k := dlySqStreams(g)
	last := len(lanes) - g
	for o := 0; ; o += k * g {
		var offs [dlyMaxStreams]int
		for t := range offs {
			offs[t] = min(o+g*min(t, k-1), last)
		}
		var acc [dlyMaxStreams][16]float32
		dlySqMulti(g, x, n, &offs, &acc)
		for t := 0; t < k; t++ {
			copy(lanes[offs[t]:offs[t]+g], acc[t][:g])
		}
		if offs[k-1] == last {
			return
		}
	}
}

// dlyX2WrappedMulti is dlyX2WrappedRef with dlySqStreams(g) windows
// accumulated at a time. A window is two stretches of whole groups, up to
// the end of the buffer and from its start, that feed its accumulators in
// that order. The stretches of a batch are cut at each other's ends, so that
// a call of dlySqMulti advances all of them by one length.
func dlyX2WrappedMulti(g int, x []float32, hSize, lo, r int, out *[dlyX2Run]float32) {
	streams := dlySqStreams(g)
	for t0 := 0; t0 < r; t0 += streams {
		k := min(streams, r-t0)
		var acc [dlyMaxStreams][16]float32
		var pos, rem [dlyMaxStreams][2]int
		var seg [dlyMaxStreams]int
		for t := range pos {
			// Windows beyond the last one repeat it. Their sums are not
			// used.
			a := lo + t0 + min(t, k-1)
			chunk1 := min(hSize, len(x)-a)
			pos[t] = [2]int{a, 0}
			rem[t] = [2]int{chunk1 &^ (g - 1), (hSize - chunk1) &^ (g - 1)}
		}
		for {
			n := hSize
			for t := range seg {
				for seg[t] < 2 && rem[t][seg[t]] == 0 {
					seg[t]++
				}
				if seg[t] == 2 {
					n = 0
					break
				}
				n = min(n, rem[t][seg[t]])
			}
			if n == 0 {
				break
			}
			var offs [dlyMaxStreams]int
			for t := range offs {
				offs[t] = pos[t][seg[t]]
				pos[t][seg[t]] += n
				rem[t][seg[t]] -= n
			}
			dlySqMulti(g, x, n, &offs, &acc)
		}
		for t := 0; t < k; t++ {
			// What is left of a window once another one has ended.
			for ; seg[t] < 2; seg[t]++ {
				p, n := pos[t][seg[t]], rem[t][seg[t]]
				dlySq(g, x[p:p+n], &acc[t])
			}
			out[t0+t] = dlyReduceX2(g, false, &acc[t], dlyX2Tails(g, x, hSize, lo+t0+t))
		}
	}
}

// dlyX2Tails returns the scalar part of the x*x sum of the window of hSize
// samples at a: the samples of each chunk beyond its whole groups, in order.
func dlyX2Tails(g int, x []float32, hSize, a int) float32 {
	chunk1 := min(hSize, len(x)-a)
	var tail float32
	for k := chunk1 &^ (g - 1); k < chunk1; k++ {
		xk := x[a+k]
		tail += float32(xk * xk)
	}
	for k := chunk1 + (hSize-chunk1)&^(g-1); k < hSize; k++ {
		xk := x[k-chunk1]
		tail += float32(xk * xk)
	}
	return tail
}

// dlyAxpyRun is a stretch [start, end) of the filter whose update reads
// x[k+off] and uses a fused (fma) or unfused multiply-add.
type dlyAxpyRun struct {
	start, end, off int
	fma             bool
}

// dlyPendingAxpy is the filter update of one capture sample, h[k] +=
// alpha*x[(xStartIndex+k) mod len(x)], split into the stretches of the C++
// lane loop.
type dlyPendingAxpy struct {
	runs   [4]dlyAxpyRun
	n      int
	alpha  float32
	active bool
}

func (p *dlyPendingAxpy) set(xStartIndex, chunk1, hSize int, alpha float32, fma bool) {
	p.n = 0
	p.alpha = alpha
	p.active = true
	add := func(start, end, off int, fma bool) {
		if start < end {
			p.runs[p.n] = dlyAxpyRun{start, end, off, fma}
			p.n++
		}
	}
	if fma {
		// AVX2: fused over the whole groups of 8 of each chunk, unfused
		// tails.
		chunk2 := hSize - chunk1
		add(0, chunk1&^7, xStartIndex, true)
		add(chunk1&^7, chunk1, xStartIndex, false)
		add(chunk1, chunk1+chunk2&^7, -chunk1, true)
		add(chunk1+chunk2&^7, hSize, -chunk1, false)
	} else {
		add(0, chunk1, xStartIndex, false)
		add(chunk1, hSize, -chunk1, false)
	}
}

// apply runs the update on h[a:b].
func (p *dlyPendingAxpy) apply(h, x []float32, a, b int) {
	for i := 0; i < p.n; i++ {
		r := &p.runs[i]
		lo, hi := max(a, r.start), min(b, r.end)
		if lo >= hi {
			continue
		}
		if r.fma {
			dlyAxpyFma(h[lo:hi], x[lo+r.off:hi+r.off], p.alpha)
		} else {
			dlyAxpy(h[lo:hi], x[lo+r.off:hi+r.off], p.alpha)
		}
	}
}

// dlyPiece is a stretch h[k:k+n] of whole groups whose h*x goes to the
// vector accumulators against x[xb:xb+n]. With upd, the filter update of the
// lane against x[xa:xa+n] is fused into it.
type dlyPiece struct {
	k, n, xb, xa int
	upd          bool
}

// dlyLaneFilter is one filter in a call of a lane core.
type dlyLaneFilter struct {
	h      []float32
	xStart int
	// sq is scratch of dlyX2SqScratch(len(h)) values for the x*x sums from a
	// table of squares, or nil.
	sq []float32
	// xw is all of x widened to float64, from which the AVX2 lane reads its
	// whole groups, or nil.
	xw []float64
	// skipIdle lets the core leave out the h*x sums of a call in which no
	// sample updates the filter (idle-filter skip). errorSum is then not
	// computed, and wasIdle is set.
	skipIdle, wasIdle bool
	errorSum          float32
	updated           bool

	// The update of the previous sample, not yet applied: pend.alpha and
	// pend.active, the start index and the first chunk of the sample, and
	// whether pend.runs is set.
	pend       dlyPendingAxpy
	pendRuns   bool
	pendStart  int
	pendChunk1 int

	// x2[a-x2Lo] is the x*x sum of the sample with start index a, for x2Lo
	// <= a < x2Hi.
	x2         [dlyX2Run]float32
	x2Lo, x2Hi int

	// The current sample: its x*x sum, the scalar tail of h*x, the vector
	// accumulators and the stretches that feed them, in order.
	x2Sum, sTail float32
	s            float32
	acc          [16]float32
	np           int
	pieces       [16]dlyPiece
}

func (f *dlyLaneFilter) addPiece(k, n, xb, xa int, upd bool) {
	f.pieces[f.np] = dlyPiece{k, n, xb, xa, upd}
	f.np++
}

// addRange lists h[p0:q], whole groups read against x[p0+off:q+off]. The
// pending update of a group runs first: fused into the stretch where a run
// of the main update kind of the lane covers whole groups, as a separate
// pass elsewhere.
func (f *dlyLaneFilter) addRange(g int, x []float32, p0, q, off int) {
	if p0 >= q {
		return
	}
	pend := &f.pend
	if !pend.active {
		f.addPiece(p0, q-p0, p0+off, 0, false)
		return
	}
	fusedFma := g == 16
	for k := p0; k < q; {
		r := &pend.runs[0]
		for i := 1; i < pend.n && pend.runs[i].start <= k; i++ {
			r = &pend.runs[i]
		}
		n := (min(r.end, q) - k) &^ (g - 1)
		if n > 0 && r.fma == fusedFma {
			f.addPiece(k, n, k+off, k+r.off, true)
			k += n
			continue
		}
		if n == 0 {
			// The group straddles two update runs.
			n = g
		}
		pend.apply(f.h, x, k, k+n)
		f.addPiece(k, n, k+off, 0, false)
		k += n
	}
}

// x2Of returns the x*x sum of the current sample, from the table that holds
// it or from a new one for the run of samples that starts with it. left is
// the number of samples left in the call, this one included.
func (f *dlyLaneFilter) x2Of(g int, x []float32, left int) float32 {
	hSize, xSize, xs := len(f.h), len(x), f.xStart
	if xs+hSize <= xSize && hSize&(g-1) == 0 {
		if lo, r, ok := f.x2TableRun(g, x, left); ok {
			if f.xw != nil {
				dlyX2SumsWide(f.xw[lo:], hSize, r, &f.x2)
			} else if f.sq != nil && g == 4 {
				// Eight chains of additions instead of four of products and
				// additions. The SSE2 lane has eight chains either way.
				dlyX2SumsSq(g, x[lo:], hSize, r, f.sq, &f.x2)
			} else {
				dlyX2Sums(g, false, x[lo:], hSize, r, &f.x2)
			}
			f.x2Lo, f.x2Hi = lo, lo+r
		}
		return f.x2[xs-f.x2Lo]
	}
	if xs < f.x2Lo || xs >= f.x2Hi {
		r := min(left, xs+1, dlyX2Run)
		lo := xs + 1 - r
		if f.xw != nil {
			dlyX2WrappedWide(x, f.xw, hSize, lo, r, &f.x2)
		} else if f.sq != nil && hSize+r-1 <= xSize {
			dlyX2WrappedSq(g, x, hSize, lo, r, f.sq, &f.x2)
		} else {
			dlyX2Wrapped(g, x, hSize, lo, r, &f.x2)
		}
		f.x2Lo, f.x2Hi = lo, xs+1
	}
	return f.x2[xs-f.x2Lo]
}

// idle reports whether no capture sample of y updates the filter: an update
// needs an x*x sum above the threshold and an unsaturated capture sample.
// The x*x sums it computes stay in the table of the filter.
func (f *dlyLaneFilter) idle(g int, x2SumThreshold float32, x, y []float32) bool {
	xs := f.xStart
	idle := true
	for i, yi := range y {
		if f.x2Of(g, x, len(y)-i) > x2SumThreshold && !dlySaturated(yi) {
			idle = false
			break
		}
		if f.xStart > 0 {
			f.xStart--
		} else {
			f.xStart = len(x) - 1
		}
	}
	f.xStart = xs
	return idle
}

// prepare computes the x*x sum of the current sample and lists its h*x work.
// left is the number of samples left in the call, this one included.
func (f *dlyLaneFilter) prepare(g int, x []float32, left int) {
	h := f.h
	hSize, xSize, xs := len(h), len(x), f.xStart
	chunk1 := min(hSize, xSize-xs)
	f.np = 0
	f.sTail = 0
	f.x2Sum = f.x2Of(g, x, left)
	if chunk1 == hSize && hSize&(g-1) == 0 {
		if !f.pend.active {
			f.addPiece(0, hSize, xs, 0, false)
			return
		}
		if f.pendChunk1 == hSize {
			f.addPiece(0, hSize, xs, f.pendStart, true)
			return
		}
	}
	chunk2 := hSize - chunk1
	v1 := chunk1 &^ (g - 1)
	v2 := chunk2 &^ (g - 1)
	pend := &f.pend
	if pend.active && !f.pendRuns {
		pend.set(f.pendStart, f.pendChunk1, hSize, pend.alpha, g == 16)
		f.pendRuns = true
	}
	var s float32
	f.addRange(g, x, 0, v1, xs)
	if pend.active {
		pend.apply(h, x, v1, chunk1)
	}
	for k := v1; k < chunk1; k++ {
		s += float32(h[k] * x[xs+k])
	}
	f.addRange(g, x, chunk1, chunk1+v2, -chunk1)
	if pend.active {
		pend.apply(h, x, chunk1+v2, hSize)
	}
	for k := chunk1 + v2; k < hSize; k++ {
		s += float32(h[k] * x[k-chunk1])
	}
	f.sTail = s
}

// run computes the filter output of the current sample from the listed
// stretches.
func (f *dlyLaneFilter) run(g int, x []float32) {
	h := f.h
	if xw := f.xw; xw != nil {
		dlyZeroAcc(&f.acc)
		for i := 0; i < f.np; i++ {
			p := &f.pieces[i]
			hk, xb := h[p.k:p.k+p.n], xw[p.xb:p.xb+p.n]
			if p.upd {
				dlyAxpyFmaDotSAvx2Wide(hk, xw[p.xa:p.xa+p.n], xb, f.pend.alpha, &f.acc)
			} else {
				dlyDotSAvx2Wide(hk, xb, &f.acc)
			}
		}
		f.s = dlyReduceS(16, &f.acc, f.sTail)
		return
	}
	if p := &f.pieces[0]; f.np == 1 && p.n == len(h) {
		// One stretch over all of h and no tails: the Sum kernels start from
		// zero accumulators and reduce in registers.
		xb := x[p.xb : p.xb+p.n]
		if p.upd {
			f.s = dlyAxpyDotSSum(g, h, x[p.xa:p.xa+p.n], xb, f.pend.alpha)
		} else {
			f.s = dlyDotSSum(g, h, xb)
		}
		return
	}
	dlyZeroAcc(&f.acc)
	for i := 0; i < f.np; i++ {
		p := &f.pieces[i]
		hk, xb := h[p.k:p.k+p.n], x[p.xb:p.xb+p.n]
		if p.upd {
			dlyAxpyDotS(g, hk, x[p.xa:p.xa+p.n], xb, f.pend.alpha, &f.acc)
		} else {
			dlyDotS(g, hk, xb, &f.acc)
		}
	}
	f.s = dlyReduceS(g, &f.acc, f.sTail)
}

// dlyRunPairNeon is run for two filters of the NEON lane. It walks the
// stretches of both in step and hands equal lengths to the two-filter kernel
// of their kinds. The stretches of a filter reach its accumulators in order.
func dlyRunPairNeon(fa, fb *dlyLaneFilter, x []float32) {
	accA := (*[4]float32)(fa.acc[:4])
	accB := (*[4]float32)(fb.acc[:4])
	*accA = [4]float32{}
	*accB = [4]float32{}
	ia, ib := 0, 0
	oa, ob := 0, 0 // consumed part of the current stretch of each filter
	for ia < fa.np && ib < fb.np {
		pa, pb := &fa.pieces[ia], &fb.pieces[ib]
		n := min(pa.n-oa, pb.n-ob)
		ha := fa.h[pa.k+oa : pa.k+oa+n]
		hb := fb.h[pb.k+ob : pb.k+ob+n]
		xbA := x[pa.xb+oa : pa.xb+oa+n]
		xbB := x[pb.xb+ob : pb.xb+ob+n]
		switch {
		case pa.upd && pb.upd:
			dlyAxpyDotS2Neon(ha, x[pa.xa+oa:pa.xa+oa+n], xbA, fa.pend.alpha, accA,
				hb, x[pb.xa+ob:pb.xa+ob+n], xbB, fb.pend.alpha, accB)
		case pa.upd:
			dlyAxpyDotDotS2Neon(ha, x[pa.xa+oa:pa.xa+oa+n], xbA, fa.pend.alpha, accA, hb, xbB, accB)
		case pb.upd:
			dlyAxpyDotDotS2Neon(hb, x[pb.xa+ob:pb.xa+ob+n], xbB, fb.pend.alpha, accB, ha, xbA, accA)
		default:
			dlyDotS2Neon(ha, xbA, accA, hb, xbB, accB)
		}
		if oa += n; oa == pa.n {
			ia, oa = ia+1, 0
		}
		if ob += n; ob == pb.n {
			ib, ob = ib+1, 0
		}
	}
	for ; ia < fa.np; ia, oa = ia+1, 0 {
		p := &fa.pieces[ia]
		hk, xb := fa.h[p.k+oa:p.k+p.n], x[p.xb+oa:p.xb+p.n]
		if p.upd {
			dlyAxpyDotSNeon(hk, x[p.xa+oa:p.xa+p.n], xb, fa.pend.alpha, accA)
		} else {
			dlyDotSNeon(hk, xb, accA)
		}
	}
	for ; ib < fb.np; ib, ob = ib+1, 0 {
		p := &fb.pieces[ib]
		hk, xb := fb.h[p.k+ob:p.k+p.n], x[p.xb+ob:p.xb+p.n]
		if p.upd {
			dlyAxpyDotSNeon(hk, x[p.xa+ob:p.xa+p.n], xb, fb.pend.alpha, accB)
		} else {
			dlyDotSNeon(hk, xb, accB)
		}
	}
	fa.s = dlyReduceS(4, &fa.acc, fa.sTail)
	fb.s = dlyReduceS(4, &fb.acc, fb.sTail)
}

// finish computes the error and the pending update from the filter output of
// the current sample and moves to the next sample.
func (f *dlyLaneFilter) finish(x2SumThreshold, smoothing, y float32, xSize int) {
	e := y - f.s
	f.errorSum += float32(e * e)
	f.pend.active = false
	f.pendRuns = false
	if f.x2Sum > x2SumThreshold && !dlySaturated(y) {
		f.pend.alpha = float32(smoothing*e) / f.x2Sum
		f.pend.active = true
		f.pendStart = f.xStart
		f.pendChunk1 = min(len(f.h), xSize-f.xStart)
		f.updated = true
	}
	if f.xStart > 0 {
		f.xStart--
	} else {
		f.xStart = xSize - 1
	}
}

// flush applies the update still pending after the last sample.
func (f *dlyLaneFilter) flush(g int, x []float32) {
	if !f.pend.active {
		return
	}
	f.pend.set(f.pendStart, f.pendChunk1, len(f.h), f.pend.alpha, g == 16)
	f.pend.apply(f.h, x, 0, len(f.h))
	f.pend.active = false
}

// dlyCoreLanes runs the MatchedFilterCore loop of the lane with group size g
// for the filter fa and, if fb is not nil, for fb in step with it.
func dlyCoreLanes(g int, fa, fb *dlyLaneFilter, x2SumThreshold, smoothing float32, x, y []float32) {
	if fa.skipIdle || (fb != nil && fb.skipIdle) {
		if fb != nil {
			dlyX2Tables2(g, fa, fb, x, len(y))
		}
		fa.wasIdle = fa.skipIdle && fa.idle(g, x2SumThreshold, x, y)
		if fb != nil {
			fb.wasIdle = fb.skipIdle && fb.idle(g, x2SumThreshold, x, y)
		}
		switch {
		case fa.wasIdle && (fb == nil || fb.wasIdle):
			return
		case fa.wasIdle:
			fa, fb = fb, nil
		case fb != nil && fb.wasIdle:
			fb = nil
		}
	}
	xSize := len(x)
	pair := fb != nil && g == 4 && dlyPairNeon
	if fa.direct(g, x, len(y)) && (fb == nil || fb.direct(g, x, len(y))) {
		dlyCoreDirect(g, fa, fb, pair, x2SumThreshold, smoothing, x, y)
		return
	}
	for i, yi := range y {
		left := len(y) - i
		if fb != nil {
			dlyX2Tables2(g, fa, fb, x, left)
		}
		fa.prepare(g, x, left)
		if fb != nil {
			fb.prepare(g, x, left)
		}
		switch {
		case pair:
			dlyRunPairNeon(fa, fb, x)
		case fb != nil:
			fa.run(g, x)
			fb.run(g, x)
		default:
			fa.run(g, x)
		}
		fa.finish(x2SumThreshold, smoothing, yi, xSize)
		if fb != nil {
			fb.finish(x2SumThreshold, smoothing, yi, xSize)
		}
	}
	fa.flush(g, x)
	if fb != nil {
		fb.flush(g, x)
	}
}

// direct reports whether no window of the next ny samples wraps and the
// filter is a whole number of groups. A sample is then one stretch over all
// of h, with the update of the sample before it fused in.
func (f *dlyLaneFilter) direct(g int, x []float32, ny int) bool {
	hSize := len(f.h)
	return hSize > 0 && hSize&(g-1) == 0 && f.xStart-(ny-1) >= 0 && f.xStart+hSize <= len(x) && !f.pend.active
}

// dlyCoreDirect is the loop of dlyCoreLanes for filters that are direct for
// all of y: the same kernels on the same stretches, without the lists of
// prepare and run.
func dlyCoreDirect(g int, fa, fb *dlyLaneFilter, pair bool, x2SumThreshold, smoothing float32, x, y []float32) {
	xSize := len(x)
	hSize := len(fa.h)
	if fb != nil && len(fb.h) != hSize {
		pair = false
	}
	for i, yi := range y {
		left := len(y) - i
		if fb != nil {
			dlyX2Tables2(g, fa, fb, x, left)
		}
		fa.x2Sum = fa.x2Of(g, x, left)
		if fb != nil {
			fb.x2Sum = fb.x2Of(g, x, left)
		}
		if pair {
			accA := (*[4]float32)(fa.acc[:4])
			accB := (*[4]float32)(fb.acc[:4])
			*accA = [4]float32{}
			*accB = [4]float32{}
			ha, hb := fa.h, fb.h
			xbA := x[fa.xStart : fa.xStart+hSize]
			xbB := x[fb.xStart : fb.xStart+hSize]
			switch {
			case fa.pend.active && fb.pend.active:
				dlyAxpyDotS2Neon(ha, x[fa.pendStart:fa.pendStart+hSize], xbA, fa.pend.alpha, accA,
					hb, x[fb.pendStart:fb.pendStart+hSize], xbB, fb.pend.alpha, accB)
			case fa.pend.active:
				dlyAxpyDotDotS2Neon(ha, x[fa.pendStart:fa.pendStart+hSize], xbA, fa.pend.alpha, accA, hb, xbB, accB)
			case fb.pend.active:
				dlyAxpyDotDotS2Neon(hb, x[fb.pendStart:fb.pendStart+hSize], xbB, fb.pend.alpha, accB, ha, xbA, accA)
			default:
				dlyDotS2Neon(ha, xbA, accA, hb, xbB, accB)
			}
			fa.s = dlyReduceS(4, &fa.acc, 0)
			fb.s = dlyReduceS(4, &fb.acc, 0)
		} else {
			fa.directSum(g, x)
			if fb != nil {
				fb.directSum(g, x)
			}
		}
		fa.finish(x2SumThreshold, smoothing, yi, xSize)
		if fb != nil {
			fb.finish(x2SumThreshold, smoothing, yi, xSize)
		}
	}
	fa.flush(g, x)
	if fb != nil {
		fb.flush(g, x)
	}
}

// directSum computes the filter output of the current sample of a direct
// filter.
func (f *dlyLaneFilter) directSum(g int, x []float32) {
	h := f.h
	if xw := f.xw; xw != nil {
		dlyZeroAcc(&f.acc)
		xb := xw[f.xStart : f.xStart+len(h)]
		if f.pend.active {
			dlyAxpyFmaDotSAvx2Wide(h, xw[f.pendStart:f.pendStart+len(h)], xb, f.pend.alpha, &f.acc)
		} else {
			dlyDotSAvx2Wide(h, xb, &f.acc)
		}
		f.s = dlyReduceS(16, &f.acc, 0)
		return
	}
	xb := x[f.xStart : f.xStart+len(h)]
	if f.pend.active {
		f.s = dlyAxpyDotSSum(g, h, x[f.pendStart:f.pendStart+len(h)], xb, f.pend.alpha)
	} else {
		f.s = dlyDotSSum(g, h, xb)
	}
}

// dlyCoreLane runs the MatchedFilterCore loop of the lane with group size g
// on one filter.
func dlyCoreLane(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	if dlyCoreLaneKernel(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum) {
		return
	}
	dlyCoreLaneSq(g, xStartIndex, x2SumThreshold, smoothing, x, nil, y, h, filtersUpdated, errorSum)
}

// dlyCoreLaneSq is dlyCoreLane through the pure-Go references. With sq,
// dlyX2SqScratch(len(h)) values of scratch, the unfused lanes compute their
// x*x sums from a table of squares.
func dlyCoreLaneSq(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, sq, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	dlyCoreLaneSkip(g, xStartIndex, x2SumThreshold, smoothing, x, nil, sq, y, h, filtersUpdated, errorSum, false)
}

// dlyCoreLaneSkip is dlyCoreLaneSq. With skipIdle it leaves errorSum alone
// when no sample updates the filter and reports that. With xw, all of x
// widened to float64, the AVX2 lane reads its whole groups from it.
func dlyCoreLaneSkip(g int, xStartIndex int, x2SumThreshold, smoothing float32, x []float32, xw []float64,
	sq, y, h []float32, filtersUpdated *bool, errorSum *float32, skipIdle bool) bool {
	var f dlyLaneFilter
	f.h, f.xStart, f.errorSum = h, xStartIndex, *errorSum
	f.skipIdle = skipIdle
	if g == 16 && len(xw) == len(x) && len(x) > 0 {
		f.xw = xw
	}
	if g != 16 && len(sq) >= dlyX2SqScratch(len(h)) {
		f.sq = sq
	}
	dlyCoreLanes(g, &f, nil, x2SumThreshold, smoothing, x, y)
	*errorSum = f.errorSum
	if f.updated {
		*filtersUpdated = true
	}
	return f.wasIdle
}

// dlyAccErrCore runs the accumulated-error core of the lane with group size
// g: MatchedFilterCoreWithAccumulatedError_NEON,
// MatchedFilterCore_AccumulatedError_SSE2 and _AVX2.
//
// The C++ copies a window that wraps into scratch memory. Here the windows
// of all ny samples of the call, each one element from the next, come from
// one stretch lin with lin[j] = x[(lo+j) mod len(x)], where lo is the start
// index of the last sample: sample i reads lin[ny-1-i:ny-1-i+len(h)]. lin is
// x itself when the stretch does not wrap and a copy in scratch, len(h)+ny-1
// samples, otherwise. The x*x sums come from dlyX2Sums, and the update of a
// sample is fused into the filter loop of the next one.
//
// A filter length that is not a multiple of g, which the C++ rejects, and a
// scratch too small for the copy run dlyAccErrCoreSeparate.
func dlyAccErrCore(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, accumulatedError, scratch []float32) {
	hSize, xSize, ny := len(h), len(x), len(y)
	if ny == 0 {
		clear(accumulatedError)
		return
	}
	span := hSize + ny - 1
	lo := ((xStartIndex-(ny-1))%xSize + xSize) % xSize
	if hSize&(g-1) != 0 || (lo+span > xSize && len(scratch) < span) {
		dlyAccErrCoreSeparate(g, xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			accumulatedError, scratch)
		return
	}
	clear(accumulatedError)
	var lin []float32
	if lo+span <= xSize {
		lin = x[lo : lo+span]
	} else {
		lin = scratch[:span]
		for n, p := 0, lo; n < span; p = 0 {
			n += copy(lin[n:], x[p:])
		}
	}
	var x2 [dlyX2Run]float32
	x2Base := -1 // x2 covers the windows at lin[x2Base:], up to dlyX2Run of them
	pending := false
	var alpha float32
	for i, yi := range y {
		j := ny - 1 - i
		if base := j &^ (dlyX2Run - 1); base != x2Base {
			r := min(dlyX2Run, ny-base)
			dlyX2Sums(g, true, lin[base:], hSize, r, &x2)
			x2Base = base
		}
		x2Sum := x2[j-x2Base]
		xb := lin[j : j+hSize]
		var s float32
		if pending {
			s = dlyAxpyAccErr(g, h, lin[j+1:j+1+hSize], xb, alpha, yi, accumulatedError)
		} else {
			s = dlyAccErr(g, h, xb, yi, accumulatedError)
		}
		e := yi - s
		*errorSum += float32(e * e)
		pending = false
		if x2Sum > x2SumThreshold && !dlySaturated(yi) {
			alpha = float32(smoothing*e) / x2Sum
			pending = true
			*filtersUpdated = true
		}
	}
	if pending {
		dlyAxpyLane(g, h, lin[:hSize], alpha)
	}
}

// dlyAccErr runs the accumulated-error filter loop of the lane for one
// sample.
func dlyAccErr(g int, h, x []float32, y float32, accErr []float32) float32 {
	switch g {
	case 4:
		return dlyAccErrNeon(h, x, y, accErr)
	case 8:
		return dlyAccErrSse2(h, x, y, accErr)
	}
	return dlyAccErrAvx2(h, x, y, accErr)
}

// dlyAxpyAccErr is dlyAccErr with the filter update of the lane against xa
// fused in front of every group.
func dlyAxpyAccErr(g int, h, xa, xb []float32, alpha, y float32, accErr []float32) float32 {
	switch g {
	case 4:
		return dlyAxpyAccErrNeon(h, xa, xb, alpha, y, accErr)
	case 8:
		return dlyAxpyAccErrSse2(h, xa, xb, alpha, y, accErr)
	}
	return dlyAxpyFmaAccErrAvx2(h, xa, xb, alpha, y, accErr)
}

// dlyAxpyLane is the filter update of the accumulated-error core of the
// lane.
func dlyAxpyLane(g int, h, x []float32, alpha float32) {
	if g == 16 {
		dlyAxpyFma(h, x, alpha)
		return
	}
	dlyAxpy(h, x, alpha)
}

// dlyAccErrCoreSeparate is the accumulated-error core as the C++ lays it
// out: per sample, a window that wraps is copied into scratch, len(h)
// samples, and the filter update is a separate pass.
func dlyAccErrCoreSeparate(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, accumulatedError, scratch []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	groups := hSize &^ (g - 1)
	// The NEON and SSE2 cores update four elements at a time, the AVX2 core
	// eight.
	updated := hSize &^ 3
	if g == 16 {
		updated = hSize &^ 7
	}
	for _, yi := range y {
		chunk1 := min(hSize, xSize-xStartIndex)
		var xp []float32
		if chunk1 != hSize {
			copy(scratch, x[xStartIndex:])
			copy(scratch[chunk1:], x[:hSize-chunk1])
			xp = scratch[:hSize]
		} else {
			xp = x[xStartIndex : xStartIndex+hSize]
		}
		var q [16]float32
		dlySq(g, xp[:groups], &q)
		x2Sum := dlyReduceX2(g, true, &q, 0)
		s := dlyAccErr(g, h[:groups], xp[:groups], yi, accumulatedError)
		e := yi - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(yi) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpyLane(g, h[:updated], xp, alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MatchedFilterCore mirrors aec3::MatchedFilterCore.
func MatchedFilterCore(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError []float32) {
	if computeAccumulatedError {
		clear(accumulatedError)
	}
	xSize := len(x)
	for i := range y {
		var x2Sum, s float32
		xIndex := xStartIndex
		if computeAccumulatedError {
			for k := range h {
				x2Sum += float32(x[xIndex] * x[xIndex])
				s += float32(h[k] * x[xIndex])
				if xIndex < xSize-1 {
					xIndex++
				} else {
					xIndex = 0
				}
				if (k+1)&0b11 == 0 {
					idx := k >> 2
					accumulatedError[idx] += float32((y[i] - s) * (y[i] - s))
				}
			}
		} else {
			for k := range h {
				x2Sum += float32(x[xIndex] * x[xIndex])
				s += float32(h[k] * x[xIndex])
				if xIndex < xSize-1 {
					xIndex++
				} else {
					xIndex = 0
				}
			}
		}

		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			xIndex2 := xStartIndex
			for k := range h {
				h[k] += float32(alpha * x[xIndex2])
				if xIndex2 < xSize-1 {
					xIndex2++
				} else {
					xIndex2 = 0
				}
			}
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MaxSquarePeakIndex returns the index of the largest square in h.
func MaxSquarePeakIndex(h []float32) int { return dlyMaxSquarePeakIndex(h) }

// dlyMaxSquarePeakIndexRef is MaxSquarePeakIndex as in the C++: it scans the
// even and the odd indices separately with strict comparisons, so each scan
// keeps its earliest maximum.
func dlyMaxSquarePeakIndexRef(h []float32) int {
	if len(h) < 2 {
		return 0
	}
	maxElement1 := h[0] * h[0]
	maxElement2 := h[1] * h[1]
	lagEstimate1 := 0
	lagEstimate2 := 1
	lastIndex := len(h) - 1
	for k := 2; k < lastIndex; k += 2 {
		element1 := h[k] * h[k]
		element2 := h[k+1] * h[k+1]
		if element1 > maxElement1 {
			maxElement1 = element1
			lagEstimate1 = k
		}
		if element2 > maxElement2 {
			maxElement2 = element2
			lagEstimate2 = k + 1
		}
	}
	if maxElement2 > maxElement1 {
		maxElement1 = maxElement2
		lagEstimate1 = lagEstimate2
	}
	// The last element of an h of odd length.
	lastElement := h[lastIndex] * h[lastIndex]
	if lastElement > maxElement1 {
		return lastIndex
	}
	return lagEstimate1
}

// MatchedFilterLagEstimate mirrors MatchedFilter::LagEstimate.
type MatchedFilterLagEstimate struct {
	Lag        int
	PreEchoLag int
}

// MatchedFilter keeps recursively updated cross-correlation estimates for
// several uniformly spaced signal shifts (matched_filter.h).
type MatchedFilter struct {
	*matchedFilterState
	// storage holds the filters, [numFilters][filterSize], and with
	// detectPreEcho the accumulated errors, [numFilters][accumulatedSize],
	// the instantaneous accumulated error and the scratch memory.
	storage []float32
}

// matchedFilterState is the state of a MatchedFilter except its storage.
type matchedFilterState struct {
	storageSpan               span
	optimization              Aec3Optimization
	subBlockSize              int
	filterIntraLagShift       int
	numFilters                int
	filterSize                int
	accumulatedSize           int // samples of one accumulated error
	scratchSize               int
	reportedLagEstimate       MatchedFilterLagEstimate
	hasReportedLagEstimate    bool
	winnerLag                 int
	hasWinnerLag              bool
	lastDetectedBestLagFilter int
	numberPreEchoUpdates      int
	excitationLimit           float32
	smoothingFast             float32
	smoothingSlow             float32
	matchingFilterThreshold   float32
	detectPreEcho             bool

	// Quiet-render skip (renderIsQuiet): quietSkip enables it,
	// quietLimitBits is the bit pattern of the largest quiet magnitude and
	// loudHint the index of a render sample that exceeded it last.
	quietSkip      bool
	quietLimitBits uint32
	loudHint       int
	// quietFilterSkips counts the cores filterIsQuiet left out.
	quietFilterSkips uint64
	// zeroCapture is set during an Update whose capture block is all ±0.
	// zeroFilterSkips counts the cores filterStaysZero left out.
	zeroCapture     bool
	zeroFilterSkips uint64
}

// NewMatchedFilter mirrors the C++ constructor.
func NewMatchedFilter(optimization Aec3Optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters,
	alignmentShiftSubBlocks int, excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold float32,
	detectPreEcho bool) *MatchedFilter {
	m := &MatchedFilter{}
	m.bind(&matchedFilterState{}, bindAlone)
	m.init(optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters, alignmentShiftSubBlocks,
		excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold, detectPreEcho)
	return m
}

func (m *MatchedFilter) bind(st *matchedFilterState, b *binder) {
	m.matchedFilterState = st
	b.f32(&m.storage, &st.storageSpan)
}

func (m *MatchedFilter) init(optimization Aec3Optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters,
	alignmentShiftSubBlocks int, excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold float32,
	detectPreEcho bool) {
	if windowSizeSubBlocks <= 0 || BlockSize%subBlockSize != 0 || subBlockSize%4 != 0 {
		panic("aec3: invalid MatchedFilter dimensions")
	}
	m.storage = nil
	*m.matchedFilterState = matchedFilterState{
		optimization:              optimization,
		subBlockSize:              subBlockSize,
		filterIntraLagShift:       alignmentShiftSubBlocks * subBlockSize,
		numFilters:                numMatchedFilters,
		filterSize:                windowSizeSubBlocks * subBlockSize,
		lastDetectedBestLagFilter: -1,
		excitationLimit:           excitationLimit,
		smoothingFast:             smoothingFast,
		smoothingSlow:             smoothingSlow,
		matchingFilterThreshold:   matchingFilterThreshold,
		detectPreEcho:             detectPreEcho,
	}
	filterSize := m.filterSize
	n := numMatchedFilters * filterSize
	if detectPreEcho {
		m.accumulatedSize = filterSize / dlyAccumulatedErrorSubSampleRate
		// One window per sample of a sub-block (dlyAccErrCore).
		m.scratchSize = filterSize + subBlockSize
		n += (numMatchedFilters+1)*m.accumulatedSize + m.scratchSize
	}
	// The scalar lane runs its filters in step. The unfused lanes compute
	// x*x from a table of squares where they run in pure Go.
	need := 0
	switch {
	case optimization == OptNone:
		need = dlyScalarScratch(filterSize)
	case dlyLanesPureGo && optimization != OptAvx2:
		need = dlyX2SqScratch(filterSize)
	}
	if extra := need - m.scratchSize; extra > 0 {
		m.scratchSize += extra
		n += extra
	}
	m.storage = make([]float32, n)
	m.resetAccumulatedErrors()
	var quietLimit float32
	quietLimit, m.quietSkip = MatchedFilterQuietLimit(filterSize, excitationLimit)
	m.quietLimitBits = math.Float32bits(quietLimit)
}

// filter returns the coefficients of filter n.
func (m *MatchedFilter) filter(n int) []float32 {
	i := n * m.filterSize
	return m.storage[i : i+m.filterSize : i+m.filterSize]
}

// accumulatedError returns the accumulated error of filter n, which exists
// with detectPreEcho.
func (m *MatchedFilter) accumulatedError(n int) []float32 {
	i := m.numFilters*m.filterSize + n*m.accumulatedSize
	return m.storage[i : i+m.accumulatedSize : i+m.accumulatedSize]
}

// instantaneousAccumulatedError returns the accumulated error of the last
// block of the filter that computes it. It is empty without detectPreEcho.
func (m *MatchedFilter) instantaneousAccumulatedError() []float32 {
	i := m.numFilters * (m.filterSize + m.accumulatedSize)
	return m.storage[i : i+m.accumulatedSize : i+m.accumulatedSize]
}

// scratchMemory returns the scratch of the accumulated-error cores, of the
// scalar lane and of the tables of squares.
func (m *MatchedFilter) scratchMemory() []float32 {
	i := m.numFilters*(m.filterSize+m.accumulatedSize) + m.accumulatedSize
	return m.storage[i : i+m.scratchSize : i+m.scratchSize]
}

// resetAccumulatedErrors sets the accumulated error of every filter to 1.
func (m *MatchedFilter) resetAccumulatedErrors() {
	if !m.detectPreEcho {
		return
	}
	i := m.numFilters * m.filterSize
	e := m.storage[i : i+m.numFilters*m.accumulatedSize]
	for k := range e {
		e[k] = 1
	}
}

// Quiet-render skip.
//
// Update returns right after clearing winner_lag and reported_lag_estimate
// when x2_sum <= x2_sum_threshold holds for every sample of every filter.
//
// A core changes state only through its filter update, which needs x2_sum >
// x2_sum_threshold. A block without an update has no winner: Update clears
// the two estimates and leaves the filters, accumulated_error,
// last_detected_best_lag_filter and number_pre_echo_updates alone.
// instantaneous_accumulated_error and scratch_memory are written before they
// are read within one call, so a skipped block leaves nothing a later block
// observes.
//
// Bound on x2_sum. Let N be the filter length and |x[k]| <= m, finite, for
// every sample of the render buffer. In every lane x2_sum is an expression
// over the N samples of the window, each used once, of rounded products
// R(x*x), rounded sums R(a+b) and, in the AVX2 lane, fused R(x*x+c), from
// accumulators that start at +0. With u = 2^-24 and eta = 2^-150, R(t) <=
// t*(1+u) + eta for a product or a fused term t >= 0, and R(a+b) <=
// (a+b)*(1+u). By induction a node over n samples whose longest path has d
// rounded operations is at most (sum of its x[k]^2 + n*eta) * (1+u)^d. No
// path is longer than N+40: a product, at most N-1 chained additions, the
// additions to the +0 accumulators and at most 5 levels of reduction. So
//
//	x2_sum <= (N*m^2 + N*eta) * (1+u)^(N+40).
//
// dlyQuietLimit takes m with N*m^2*(1+margin) <= x2_sum_threshold for margin
// = 4*(N+40)*u, twice the 2*(N+40)*u that bounds (1+u)^(N+40) - 1 for N <=
// 2^16. The other half covers N*eta, which is below x2_sum_threshold*2^-66
// for the thresholds, at least 2^-60, that get a nonzero m. A smaller
// threshold gets m = 0: a buffer of zeros has x2_sum = 0, which exceeds no
// threshold >= 0. A threshold that is negative, NaN or infinite, or N >
// 2^16, disables the skip.
//
// The test |x[k]| <= m compares bit patterns without the sign bit, which
// orders magnitudes and puts Inf and NaN above every finite m.

// Quiet-filter skip.
//
// Update does not run the core of a filter whose windows hold no render
// sample above m in the block (filterIsQuiet): by the bound above the filter
// cannot update. The windows of a block span the filter length plus the
// sub-block size minus one samples.
//
// A filter that does not update leaves no trace in Update: its error sum is
// read only if it updated, and its lag is the peak of its unchanged
// coefficients. The filter that computes the accumulated error always runs:
// Update reads instantaneous_accumulated_error when that filter wins, which
// the overlap rule allows without an update of its own.

// Idle-filter skip.
//
// The cores that have the x*x sums of a block before its h*x sums (the
// pure-Go lane cores, the two-filter core of the NEON lane and the scalar
// lane in step) test the condition itself instead of a bound: a filter that
// no capture sample of the block updates does not compute its h*x sums and
// its error sum. The filter that computes the accumulated error always runs.

// Zero-filter skip.
//
// Update does not run the core of a filter that is all +0 in a block whose
// capture samples are all ±0, if the render samples in its windows are
// finite (filterStaysZero).
//
// The anchor of the error sums is +0 in such a block, and no error sum, a
// sum of squares, is below a multiple of it: the block has no winner, and a
// core matters only through the filter it updates. The output of a +0 filter
// is +0, the error y - s is ±0, and so is the step alpha =
// smoothing*e/x2_sum: an update needs x2_sum above a threshold that is not
// negative, and smoothing is finite. +0 plus ±0 times a finite sample is +0,
// fused or not.

// dlyQuietMargin returns the relative margin of the quiet limit of a filter
// of n samples.
func dlyQuietMargin(n int) float64 { return 4 * float64(n+40) / (1 << 24) }

// dlyQuietLimit returns the largest magnitude m with n*m^2*(1+margin) <=
// x2SumThreshold, rounded down, and whether the skip applies.
func dlyQuietLimit(n int, x2SumThreshold float32, margin float64) (float32, bool) {
	t := float64(x2SumThreshold)
	if !(t >= 0) || math.IsInf(t, 0) || n <= 0 || n > 1<<16 {
		return 0, false
	}
	if t < 0x1p-60 {
		return 0, true
	}
	m64 := math.Sqrt(t / (float64(n) * (1 + margin)))
	m := float32(m64)
	for float64(m) > m64 {
		m = math.Nextafter32(m, 0)
	}
	// One float32 step down covers the rounding of the float64 evaluation.
	return math.Nextafter32(m, 0), true
}

// MatchedFilterQuietLimit returns the quiet limit of a MatchedFilter with
// filters of filterSize samples: no filter updates while no sample of the
// render buffer exceeds it in magnitude. It reports false when the skip does
// not apply.
func MatchedFilterQuietLimit(filterSize int, excitationLimit float32) (float32, bool) {
	x2SumThreshold := float32(float32(filterSize)*excitationLimit) * excitationLimit
	return dlyQuietLimit(filterSize, x2SumThreshold, dlyQuietMargin(filterSize))
}

// dlyQuietSkips counts the blocks the quiet-render skip left out, in every
// MatchedFilter of the process, once a test has asked for the count.
var (
	dlyQuietSkips    atomic.Uint64
	dlyQuietCounting atomic.Bool
)

// MatchedFilterQuietSkipsForTesting returns the number of Update calls that
// took the quiet-render skip, in every MatchedFilter of the process, since
// the first call of this function.
func MatchedFilterQuietSkipsForTesting() uint64 {
	dlyQuietCounting.Store(true)
	return dlyQuietSkips.Load()
}

// QuietFilterSkipsForTesting returns the number of cores the quiet-filter
// skip left out.
func (m *MatchedFilter) QuietFilterSkipsForTesting() uint64 { return m.quietFilterSkips }

// dlyQuiet reports whether no sample of x has a magnitude above the one with
// the bit pattern limit.
func dlyQuiet(x []float32, limit uint32) bool {
	i := 0
	for ; i+8 <= len(x); i += 8 {
		xx := (*[8]float32)(x[i : i+8])
		a := max(math.Float32bits(xx[0])&0x7fffffff, math.Float32bits(xx[1])&0x7fffffff)
		b := max(math.Float32bits(xx[2])&0x7fffffff, math.Float32bits(xx[3])&0x7fffffff)
		c := max(math.Float32bits(xx[4])&0x7fffffff, math.Float32bits(xx[5])&0x7fffffff)
		d := max(math.Float32bits(xx[6])&0x7fffffff, math.Float32bits(xx[7])&0x7fffffff)
		if max(a, b, c, d) > limit {
			return false
		}
	}
	for ; i < len(x); i++ {
		if math.Float32bits(x[i])&0x7fffffff > limit {
			return false
		}
	}
	return true
}

// filterIsQuiet reports whether no render sample in the windows of filter n
// for the current block exceeds the quiet limit, and counts the skip.
func (m *MatchedFilter) filterIsQuiet(renderBuffer *DownsampledRenderBuffer, n int) bool {
	x := renderBuffer.Buffer
	// The capture samples of the block have the start indices xStart,
	// xStart-1, ..., xStart-(subBlockSize-1), around the start of the
	// buffer.
	span := m.filterSize + m.subBlockSize - 1
	if !m.quietSkip || span > len(x) {
		return false
	}
	lo := m.xStartIndex(renderBuffer, n) - (m.subBlockSize - 1)
	if lo < 0 {
		lo += len(x)
	}
	n1 := min(span, len(x)-lo)
	if !dlyQuiet(x[lo:lo+n1], m.quietLimitBits) || !dlyQuiet(x[:span-n1], m.quietLimitBits) {
		return false
	}
	m.quietFilterSkips++
	return true
}

// dlyZeroCapture reports whether every capture sample is ±0 and smoothing is
// finite.
func dlyZeroCapture(y []float32, smoothing float32) bool {
	var bits uint32
	for _, v := range y {
		bits |= math.Float32bits(v)
	}
	return bits&0x7fffffff == 0 && math.Float32bits(smoothing)&0x7fffffff < 0x7f800000
}

// filterStaysZero reports whether filter n is all +0 and the render samples
// in its windows for the current block are finite, and counts the skip.
func (m *MatchedFilter) filterStaysZero(renderBuffer *DownsampledRenderBuffer, n int) bool {
	var bits uint32
	for _, v := range m.filter(n) {
		bits |= math.Float32bits(v)
	}
	if bits != 0 {
		return false
	}
	const maxFinite = 0x7f7fffff
	x := renderBuffer.Buffer
	span := m.filterSize + m.subBlockSize - 1
	if span > len(x) {
		if !dlyQuiet(x, maxFinite) {
			return false
		}
	} else {
		lo := m.xStartIndex(renderBuffer, n) - (m.subBlockSize - 1)
		if lo < 0 {
			lo += len(x)
		}
		n1 := min(span, len(x)-lo)
		if !dlyQuiet(x[lo:lo+n1], maxFinite) || !dlyQuiet(x[:span-n1], maxFinite) {
			return false
		}
	}
	m.zeroFilterSkips++
	return true
}

// ZeroFilterSkipsForTesting returns the number of cores the zero-filter skip
// left out.
func (m *MatchedFilter) ZeroFilterSkipsForTesting() uint64 { return m.zeroFilterSkips }

// renderIsQuiet reports whether no sample of the render buffer exceeds the
// quiet limit.
func (m *MatchedFilter) renderIsQuiet(x []float32) bool {
	limit := m.quietLimitBits
	if h := m.loudHint; h < len(x) && math.Float32bits(x[h])&0x7fffffff > limit {
		return false
	}
	i := 0
	for ; i+16 <= len(x); i += 16 {
		xx := (*[16]float32)(x[i : i+16])
		var a, b, c, d uint32
		for j := 0; j < 16; j += 4 {
			a = max(a, math.Float32bits(xx[j])&0x7fffffff)
			b = max(b, math.Float32bits(xx[j+1])&0x7fffffff)
			c = max(c, math.Float32bits(xx[j+2])&0x7fffffff)
			d = max(d, math.Float32bits(xx[j+3])&0x7fffffff)
		}
		if max(a, b, c, d) > limit {
			break
		}
	}
	for ; i < len(x); i++ {
		if math.Float32bits(x[i])&0x7fffffff > limit {
			m.loudHint = i
			return false
		}
	}
	return true
}

// Reset clears the filters and the estimates. With fullReset it resets the
// pre-echo state too.
func (m *MatchedFilter) Reset(fullReset bool) {
	clear(m.storage[:m.numFilters*m.filterSize])
	m.hasWinnerLag = false
	m.hasReportedLagEstimate = false
	if fullReset {
		m.resetAccumulatedErrors()
		m.numberPreEchoUpdates = 0
	}
}

// GetBestLagEstimate returns the lag estimate, if there is one.
func (m *MatchedFilter) GetBestLagEstimate() (MatchedFilterLagEstimate, bool) {
	if !m.hasReportedLagEstimate {
		return MatchedFilterLagEstimate{}, false
	}
	return m.reportedLagEstimate, true
}

// GetMaxFilterLag returns the largest lag the filters cover.
func (m *MatchedFilter) GetMaxFilterLag() int {
	return m.numFilters*m.filterIntraLagShift + m.filterSize
}

// dlyLaneGroupSize returns the group size of a SIMD lane.
func dlyLaneGroupSize(optimization Aec3Optimization) int {
	switch optimization {
	case OptNeon:
		return 4
	case OptSse2:
		return 8
	}
	return 16
}

func (m *MatchedFilter) runCore(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computePreEcho bool) {
	m.runCoreSkip(xStartIndex, x2SumThreshold, smoothing, x, nil, y, h, filtersUpdated, errorSum, computePreEcho,
		false)
}

// runCoreSkip is runCore. With skipIdle, a core that has its x*x sums before
// its h*x sums computes neither the latter nor errorSum when no sample
// updates the filter. xw is x widened to float64 for the pure-Go core of the
// AVX2 lane, or nil.
func (m *MatchedFilter) runCoreSkip(xStartIndex int, x2SumThreshold, smoothing float32, x []float32,
	xw []float64, y, h []float32, filtersUpdated *bool, errorSum *float32, computePreEcho, skipIdle bool) {
	if dlyLanesPureGo && !computePreEcho && m.optimization != OptNone {
		// Only the accumulated-error cores use the scratch.
		if dlyCoreLaneSkip(dlyLaneGroupSize(m.optimization), xStartIndex, x2SumThreshold, smoothing, x, xw,
			m.scratchMemory(), y, h, filtersUpdated, errorSum, m.quietSkip && skipIdle) {
			m.quietFilterSkips++
		}
		return
	}
	switch m.optimization {
	case OptSse2:
		MatchedFilterCoreSse2(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError(), m.scratchMemory())
	case OptAvx2:
		MatchedFilterCoreAvx2(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError(), m.scratchMemory())
	case OptNeon:
		MatchedFilterCoreNeon(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError(), m.scratchMemory())
	default:
		MatchedFilterCore(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError())
	}
}

// dlyMaxPairedFilters is the largest number of filters whose cores Update
// runs in step.
const dlyMaxPairedFilters = 16

// xStartIndex returns the start index of filter n in the render buffer.
func (m *MatchedFilter) xStartIndex(renderBuffer *DownsampledRenderBuffer, n int) int {
	return (renderBuffer.Read + n*m.filterIntraLagShift + m.subBlockSize - 1) % len(renderBuffer.Buffer)
}

// runCoresPaired runs the NEON-lane cores of all filters for one block and
// stores their error sums and update flags. Cores share no state, so their
// order is free: the filters without accumulated error run two at a time.
func (m *MatchedFilter) runCoresPaired(renderBuffer *DownsampledRenderBuffer, y []float32, x2SumThreshold,
	smoothing float32, errorSums *[dlyMaxPairedFilters]float32, updated *[dlyMaxPairedFilters]bool) {
	x := renderBuffer.Buffer
	var fa, fb dlyLaneFilter
	first := -1
	for n := 0; n < m.numFilters; n++ {
		if m.zeroCapture && m.filterStaysZero(renderBuffer, n) {
			continue
		}
		if m.detectPreEcho && n == m.lastDetectedBestLagFilter {
			m.runCore(m.xStartIndex(renderBuffer, n), x2SumThreshold, smoothing, x, y, m.filter(n),
				&updated[n], &errorSums[n], true)
			continue
		}
		if m.filterIsQuiet(renderBuffer, n) {
			continue
		}
		if first < 0 {
			first = n
			continue
		}
		fa = dlyLaneFilter{h: m.filter(first), xStart: m.xStartIndex(renderBuffer, first), skipIdle: m.quietSkip}
		fb = dlyLaneFilter{h: m.filter(n), xStart: m.xStartIndex(renderBuffer, n), skipIdle: m.quietSkip}
		dlyCoreLanes(4, &fa, &fb, x2SumThreshold, smoothing, x, y)
		errorSums[first], updated[first] = fa.errorSum, fa.updated
		errorSums[n], updated[n] = fb.errorSum, fb.updated
		if fa.wasIdle {
			m.quietFilterSkips++
		}
		if fb.wasIdle {
			m.quietFilterSkips++
		}
		first = -1
	}
	if first >= 0 {
		m.runCoreSkip(m.xStartIndex(renderBuffer, first), x2SumThreshold, smoothing, x, nil, y, m.filter(first),
			&updated[first], &errorSums[first], false, true)
	}
}

// dlyScalarInStep runs the filters of the scalar lane in step. Tests turn it
// off to compare with MatchedFilterCore.
var dlyScalarInStep = true

// runCoresScalar runs the scalar-lane cores of all filters for one block, up
// to dlyScalarMax in step, and stores their error sums and update flags. The
// filter that computes the accumulated error leads its group.
func (m *MatchedFilter) runCoresScalar(renderBuffer *DownsampledRenderBuffer, y []float32, x2SumThreshold,
	smoothing float32, errorSums *[dlyMaxPairedFilters]float32, updated *[dlyMaxPairedFilters]bool) {
	x := renderBuffer.Buffer
	scratch := m.scratchMemory()
	var order [dlyMaxPairedFilters]int
	count := 0
	accFilter := -1
	var stays [dlyMaxPairedFilters]bool
	for n := 0; n < m.numFilters; n++ {
		stays[n] = m.zeroCapture && m.filterStaysZero(renderBuffer, n)
	}
	if n := m.lastDetectedBestLagFilter; m.detectPreEcho && n >= 0 && n < m.numFilters && !stays[n] {
		accFilter = n
		order[0] = accFilter
		count = 1
		clear(m.instantaneousAccumulatedError())
	}
	for n := 0; n < m.numFilters; n++ {
		if stays[n] {
			continue
		}
		if n != accFilter && !m.filterIsQuiet(renderBuffer, n) {
			order[count] = n
			count++
		}
	}
	for start := 0; start < count; {
		size := min(count-start, dlyScalarMax)
		if count-start == dlyScalarMax+1 {
			// Two groups of two rather than three and a single chain.
			size = 2
		}
		var fs [dlyScalarMax]dlyScalarFilter
		for i := 0; i < size; i++ {
			n := order[start+i]
			fs[i].h, fs[i].xStart = m.filter(n), m.xStartIndex(renderBuffer, n)
			if n == accFilter {
				fs[i].accErr = m.instantaneousAccumulatedError()
			}
		}
		dlyCoreScalar(fs[:size], x2SumThreshold, smoothing, x, y, scratch, m.quietSkip)
		for i := 0; i < size; i++ {
			n := order[start+i]
			errorSums[n], updated[n] = fs[i].errorSum, fs[i].updated
			m.quietFilterSkips += uint64(fs[i].idleRuns)
		}
		start += size
	}
}

// Update updates the filters with the capture samples of one sub-block and
// selects the lag estimate.
func (m *MatchedFilter) Update(renderBuffer *DownsampledRenderBuffer, capture []float32, useSlowSmoothing bool) {
	m.UpdateWide(renderBuffer, capture, useSlowSmoothing, nil)
}

// UpdateWide is Update with work space for the render samples widened to
// float64 (dlyWide). wide may be nil.
func (m *MatchedFilter) UpdateWide(renderBuffer *DownsampledRenderBuffer, capture []float32, useSlowSmoothing bool,
	wide *dlyWide) {
	y := capture[:m.subBlockSize]

	smoothing := m.smoothingFast
	if useSlowSmoothing {
		smoothing = m.smoothingSlow
	}

	x2SumThreshold := float32(float32(m.filterSize)*m.excitationLimit) * m.excitationLimit

	// The anchor of the error sums.
	var errorSumAnchor float32
	for _, v := range y {
		errorSumAnchor += float32(v * v)
	}

	winnerErrorSum := errorSumAnchor
	m.hasWinnerLag = false
	m.hasReportedLagEstimate = false
	if m.quietSkip && m.renderIsQuiet(renderBuffer.Buffer) {
		// No filter can update, so there is no winner (quiet-render skip).
		if dlyQuietCounting.Load() {
			dlyQuietSkips.Add(1)
		}
		return
	}
	m.zeroCapture = m.quietSkip && dlyZeroCapture(y, smoothing)
	var xw []float64
	if m.optimization == OptAvx2 && wide != nil {
		xw = wide.fill(renderBuffer.Buffer)
	}
	alignmentShift := 0
	// The peak of filter peakFilter and of the filter before the current
	// one, or -1 where not computed.
	peak, peakFilter := 0, -1
	previousPeak, previousPeakFilter := 0, -1
	winnerIndex := -1
	var pairedErrorSums [dlyMaxPairedFilters]float32
	var pairedUpdated [dlyMaxPairedFilters]bool
	paired := m.optimization == OptNeon && dlyPairNeon && m.numFilters <= dlyMaxPairedFilters
	if paired {
		m.runCoresPaired(renderBuffer, y, x2SumThreshold, smoothing, &pairedErrorSums, &pairedUpdated)
	}
	// The scalar lane runs its filters in step where its scratch and the
	// render buffer allow.
	if m.optimization == OptNone && dlyScalarInStep && m.numFilters <= dlyMaxPairedFilters &&
		m.filterSize+dlyX2Run <= len(renderBuffer.Buffer) && m.scratchSize >= dlyScalarScratch(m.filterSize) {
		m.runCoresScalar(renderBuffer, y, x2SumThreshold, smoothing, &pairedErrorSums, &pairedUpdated)
		paired = true
	}
	for n := 0; n < m.numFilters; n++ {
		var errorSum float32
		filtersUpdated := false
		if paired {
			errorSum, filtersUpdated = pairedErrorSums[n], pairedUpdated[n]
		} else {
			computePreEcho := m.detectPreEcho && n == m.lastDetectedBestLagFilter
			if m.zeroCapture && m.filterStaysZero(renderBuffer, n) {
				// Nothing to run.
			} else if computePreEcho || !m.filterIsQuiet(renderBuffer, n) {
				m.runCoreSkip(m.xStartIndex(renderBuffer, n), x2SumThreshold, smoothing, renderBuffer.Buffer, xw, y,
					m.filter(n), &filtersUpdated, &errorSum, computePreEcho, true)
			}
		}

		// The lag estimate of a filter is the position of its peak. The C++
		// computes it for every filter and reads it in two places: to decide
		// whether a filter that updated, with an error sum below the
		// threshold and the best so far, wins, and to compare the lag of the
		// filter before such a winner with it. The peak is a function of the
		// filter alone, so it is computed where it is read.
		if filtersUpdated && errorSum < float32(m.matchingFilterThreshold*errorSumAnchor) &&
			errorSum < winnerErrorSum {
			if peakFilter != n {
				peak, peakFilter = MaxSquarePeakIndex(m.filter(n)), n
			}
			lagEstimate := peak
			if lagEstimate > 2 && lagEstimate < m.filterSize-10 {
				lag := lagEstimate + alignmentShift
				winnerErrorSum = errorSum
				winnerIndex = n
				// Of two filters with the same winner candidate, in the
				// overlap, the one with the smaller index wins: it searches
				// for pre-echoes.
				m.winnerLag = lag
				if n > 0 {
					if previousPeakFilter != n-1 {
						previousPeak, previousPeakFilter = MaxSquarePeakIndex(m.filter(n-1)), n-1
					}
					if previousPeak+alignmentShift-m.filterIntraLagShift == lag {
						winnerIndex = n - 1
					}
				}
				m.hasWinnerLag = true
			}
		}
		previousPeak, previousPeakFilter = peak, peakFilter
		alignmentShift += m.filterIntraLagShift
	}

	if winnerIndex != -1 {
		m.reportedLagEstimate = MatchedFilterLagEstimate{Lag: m.winnerLag, PreEchoLag: m.winnerLag}
		m.hasReportedLagEstimate = true
		if m.detectPreEcho && m.lastDetectedBestLagFilter == winnerIndex {
			const energyThreshold = 1.0
			if errorSumAnchor > energyThreshold {
				dlyUpdateAccumulatedError(m.instantaneousAccumulatedError(), m.accumulatedError(winnerIndex),
					1/errorSumAnchor)
				m.numberPreEchoUpdates++
			}
			if m.numberPreEchoUpdates >= 50 {
				m.reportedLagEstimate.PreEchoLag = dlyComputePreEchoLag(m.accumulatedError(winnerIndex),
					m.winnerLag, winnerIndex*m.filterIntraLagShift)
			} else {
				m.reportedLagEstimate.PreEchoLag = m.winnerLag
			}
		}
		m.lastDetectedBestLagFilter = winnerIndex
	}
}
