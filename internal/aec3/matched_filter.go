package aec3

import (
	"math"
	"sync/atomic"
)

// Subsample rate used for computing the accumulated error. The implementation
// of some core functions depends on this constant being equal to 4.
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
// aec3::MatchedFilterCoreWithAccumulatedError_NEON.
func MatchedFilterCoreWithAccumulatedErrorNeon(xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, accumulatedError, scratchMemory []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	for i := range y {
		chunk1 := min(hSize, xSize-xStartIndex)
		var xp []float32
		if chunk1 != hSize {
			chunk2 := hSize - chunk1
			copy(scratchMemory, x[xStartIndex:])
			copy(scratchMemory[chunk1:], x[:chunk2])
			xp = scratchMemory[:hSize]
		} else {
			xp = x[xStartIndex : xStartIndex+hSize]
		}
		var x2v [4]float32
		s := dlyAccErrNeon(h, xp, y[i], accumulatedError, &x2v)
		var x2Sum float32
		x2Sum += (x2v[0] + x2v[1]) + (x2v[2] + x2v[3])
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpy(h[:hSize&^3], xp, alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MatchedFilterCoreNeon mirrors aec3::MatchedFilterCore_NEON.
func MatchedFilterCoreNeon(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		MatchedFilterCoreWithAccumulatedErrorNeon(xStartIndex, x2SumThreshold, smoothing, x, y, h,
			filtersUpdated, errorSum, accumulatedError, scratchMemory)
		return
	}
	dlyCoreLane(4, xStartIndex, x2SumThreshold,
		smoothing, x, y, h, filtersUpdated, errorSum)
}

// MatchedFilterCoreAccumulatedErrorSse2 mirrors
// aec3::MatchedFilterCore_AccumulatedError_SSE2.
func MatchedFilterCoreAccumulatedErrorSse2(xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, accumulatedError, scratchMemory []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	for i := range y {
		chunk1 := min(hSize, xSize-xStartIndex)
		var xp []float32
		if chunk1 != hSize {
			chunk2 := hSize - chunk1
			copy(scratchMemory, x[xStartIndex:])
			copy(scratchMemory[chunk1:], x[:chunk2])
			xp = scratchMemory[:hSize]
		} else {
			xp = x[xStartIndex : xStartIndex+hSize]
		}
		var x2v [8]float32
		s := dlyAccErrSse2(h, xp, y[i], accumulatedError, &x2v)
		var v [4]float32
		for l := range v {
			v[l] = x2v[l] + x2v[4+l]
		}
		var x2Sum float32
		x2Sum += ((v[0] + v[1]) + v[2]) + v[3]
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpy(h[:hSize&^3], xp, alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// MatchedFilterCoreSse2 mirrors aec3::MatchedFilterCore_SSE2.
func MatchedFilterCoreSse2(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		MatchedFilterCoreAccumulatedErrorSse2(xStartIndex, x2SumThreshold, smoothing, x, y, h,
			filtersUpdated, errorSum, accumulatedError, scratchMemory)
		return
	}
	dlyCoreLane(8, xStartIndex, x2SumThreshold,
		smoothing, x, y, h, filtersUpdated, errorSum)
}

// MatchedFilterCoreAccumulatedErrorAvx2 mirrors
// aec3::MatchedFilterCore_AccumulatedError_AVX2.
func MatchedFilterCoreAccumulatedErrorAvx2(xStartIndex int, x2SumThreshold, smoothing float32,
	x, y, h []float32, filtersUpdated *bool, errorSum *float32, accumulatedError, scratchMemory []float32) {
	hSize := len(h)
	xSize := len(x)
	clear(accumulatedError)
	for i := range y {
		chunk1 := min(hSize, xSize-xStartIndex)
		var xp []float32
		if chunk1 != hSize {
			chunk2 := hSize - chunk1
			copy(scratchMemory, x[xStartIndex:])
			copy(scratchMemory[chunk1:], x[:chunk2])
			xp = scratchMemory[:hSize]
		} else {
			xp = x[xStartIndex : xStartIndex+hSize]
		}
		s, x2Sum := dlyAccErrAvx2Sample(h, xp, y[i], accumulatedError)
		e := y[i] - s
		*errorSum += float32(e * e)
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			dlyAxpyFma(h[:hSize&^7], xp, alpha)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
}

// dlyAccErrAvx2SampleRef runs dlyAccErrAvx2Ref from zero x*x
// accumulators and returns the filter output and the x*x sum of
// MatchedFilterCore_AccumulatedError_AVX2 for one capture sample.
func dlyAccErrAvx2SampleRef(h, x []float32, y float32, accErr []float32) (s, x2Sum float32) {
	var x2v [16]float32
	s = dlyAccErrAvx2Ref(h, x, y, accErr, &x2v)
	var a [8]float32
	for l := range a {
		a[l] = x2v[l] + x2v[8+l]
	}
	var v [4]float32
	for l := range v {
		v[l] = a[l] + a[4+l]
	}
	x2Sum += ((v[0] + v[1]) + v[2]) + v[3]
	return s, x2Sum
}

// MatchedFilterCoreAvx2 mirrors aec3::MatchedFilterCore_AVX2.
func MatchedFilterCoreAvx2(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computeAccumulatedError bool, accumulatedError, scratchMemory []float32) {
	if computeAccumulatedError {
		MatchedFilterCoreAccumulatedErrorAvx2(xStartIndex, x2SumThreshold, smoothing, x, y, h,
			filtersUpdated, errorSum, accumulatedError, scratchMemory)
		return
	}
	dlyCoreLane(16, xStartIndex, x2SumThreshold,
		smoothing, x, y, h, filtersUpdated, errorSum)
}

// dlyReduceNeon adds the NEON lane sums to the scalar tail sums.
func dlyReduceNeon(acc *[8]float32, s, x2Sum float32) (float32, float32) {
	s += (acc[0] + acc[1]) + (acc[2] + acc[3])
	x2Sum += (acc[4] + acc[5]) + (acc[6] + acc[7])
	return s, x2Sum
}

// dlyReduceSse2Ref adds the SSE2 lane sums to the scalar tail sums.
func dlyReduceSse2Ref(acc *[16]float32, s, x2Sum float32) (float32, float32) {
	var v [4]float32
	for l := range v {
		v[l] = acc[8+l] + acc[12+l]
	}
	x2Sum += ((v[0] + v[1]) + v[2]) + v[3]
	for l := range v {
		v[l] = acc[l] + acc[4+l]
	}
	s += ((v[0] + v[1]) + v[2]) + v[3]
	return s, x2Sum
}

// dlyReduceAvx2Ref adds the AVX2 lane sums to the scalar tail sums:
// hsum_ab(x2_sum_256 + x2_sum_256_8, s_256 + s_256_8).
func dlyReduceAvx2Ref(acc *[32]float32, s, x2Sum float32) (float32, float32) {
	var a, b [8]float32
	for l := range a {
		a[l] = acc[16+l] + acc[24+l]
		b[l] = acc[l] + acc[8+l]
	}
	x2Sum += ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
	s += ((b[0] + b[1]) + (b[2] + b[3])) + ((b[4] + b[5]) + (b[6] + b[7]))
	return s, x2Sum
}

// dlyDotSse2SumRef is dlyDotSse2Ref from zero accumulators, reduced.
func dlyDotSse2SumRef(h, x []float32) (s, x2Sum float32) {
	var acc [16]float32
	dlyDotSse2Ref(h, x, &acc)
	return dlyReduceSse2Ref(&acc, 0, 0)
}

// dlyAxpyDotSse2SumRef is dlyAxpyDotSse2Ref from zero accumulators, reduced.
func dlyAxpyDotSse2SumRef(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	var acc [16]float32
	dlyAxpyDotSse2Ref(h, xa, xb, alpha, &acc)
	return dlyReduceSse2Ref(&acc, 0, 0)
}

// dlyDotAvx2SumRef is dlyDotAvx2Ref from zero accumulators, reduced.
func dlyDotAvx2SumRef(h, x []float32) (s, x2Sum float32) {
	var acc [32]float32
	dlyDotAvx2Ref(h, x, &acc)
	return dlyReduceAvx2Ref(&acc, 0, 0)
}

// dlyAxpyFmaDotAvx2SumRef is dlyAxpyFmaDotAvx2Ref from zero accumulators,
// reduced.
func dlyAxpyFmaDotAvx2SumRef(h, xa, xb []float32, alpha float32) (s, x2Sum float32) {
	var acc [32]float32
	dlyAxpyFmaDotAvx2Ref(h, xa, xb, alpha, &acc)
	return dlyReduceAvx2Ref(&acc, 0, 0)
}

// dlyAxpyRun is a stretch [start, end) of the filter whose update reads
// x[k+off] and uses a fused (fma) or unfused multiply-add.
type dlyAxpyRun struct {
	start, end, off int
	fma             bool
}

// dlyPendingAxpy is the filter update of one capture sample, h[k] +=
// alpha*x[(xStartIndex+k) mod len(x)], split into the stretches the C++ lane
// loop uses. The next sample's dot product applies it to each element just
// before reading that element; every element sees the same operations as
// with a separate update pass.
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
		// AVX2: fused over the multiple-of-8 part of each chunk, unfused tail.
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

// dlyLaneDot runs the dot kernel of the lane with group size g on the lane's
// accumulators, which occupy the front of acc.
func dlyLaneDot(g int, h, x []float32, acc *[32]float32) {
	switch g {
	case 4:
		dlyDotNeon(h, x, (*[8]float32)(acc[:8]))
	case 8:
		dlyDotSse2(h, x, (*[16]float32)(acc[:16]))
	default:
		dlyDotAvx2(h, x, acc)
	}
}

// dlyDotSegment accumulates h[p0:q]·x[p0+off:q+off] into the lane
// accumulators in groups of g, first applying the pending update to every
// group. Stretches of whole groups inside one update run of the lane's main
// update kind go through the fused update-and-dot kernel.
func dlyDotSegment(g int, pend *dlyPendingAxpy, h, x []float32, p0, q, off int, acc *[32]float32) {
	if !pend.active {
		dlyLaneDot(g, h[p0:q], x[p0+off:q+off], acc)
		return
	}
	fusedFma := g == 16
	k := p0
	for k < q {
		r := &pend.runs[0]
		for i := 1; i < pend.n && pend.runs[i].start <= k; i++ {
			r = &pend.runs[i]
		}
		n := (min(r.end, q) - k) &^ (g - 1)
		if n > 0 {
			if r.fma == fusedFma {
				hk, xa, xb := h[k:k+n], x[k+r.off:k+n+r.off], x[k+off:k+n+off]
				switch g {
				case 4:
					dlyAxpyDotNeon(hk, xa, xb, pend.alpha, (*[8]float32)(acc[:8]))
				case 8:
					dlyAxpyDotSse2(hk, xa, xb, pend.alpha, (*[16]float32)(acc[:16]))
				default:
					dlyAxpyFmaDotAvx2(hk, xa, xb, pend.alpha, acc)
				}
			} else {
				pend.apply(h, x, k, k+n)
				dlyLaneDot(g, h[k:k+n], x[k+off:k+n+off], acc)
			}
			k += n
			continue
		}
		// The group straddles two update runs.
		pend.apply(h, x, k, k+g)
		dlyLaneDot(g, h[k:k+g], x[k+off:k+g+off], acc)
		k += g
	}
}

// dlyCoreFused runs the MatchedFilterCore loop of the SIMD lane with group
// size g (4 NEON, 8 SSE2, 16 AVX2). Per capture sample the lane accumulates
// the multiple-of-g part of each of the two circular-buffer chunks in its
// vector accumulators and the rest in scalars, then reduces. The filter
// update of a sample is deferred into the next sample's dot product (or a
// final pass), which updates each h element right before reading it.
func dlyCoreFused(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	hSize := len(h)
	xSize := len(x)
	var pend dlyPendingAxpy
	for i := range y {
		var acc [32]float32
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1

		v1 := chunk1 &^ (g - 1)
		dlyDotSegment(g, &pend, h, x, 0, v1, xStartIndex, &acc)
		if pend.active {
			pend.apply(h, x, v1, chunk1)
		}
		for k := v1; k < chunk1; k++ {
			xk := x[xStartIndex+k]
			x2Sum += float32(xk * xk)
			s += float32(h[k] * xk)
		}
		v2 := chunk2 &^ (g - 1)
		dlyDotSegment(g, &pend, h, x, chunk1, chunk1+v2, -chunk1, &acc)
		if pend.active {
			pend.apply(h, x, chunk1+v2, hSize)
		}
		for k := chunk1 + v2; k < hSize; k++ {
			xk := x[k-chunk1]
			x2Sum += float32(xk * xk)
			s += float32(h[k] * xk)
		}
		switch g {
		case 4:
			s, x2Sum = dlyReduceNeon((*[8]float32)(acc[:8]), s, x2Sum)
		case 8:
			s, x2Sum = dlyReduceSse2((*[16]float32)(acc[:16]), s, x2Sum)
		default:
			s, x2Sum = dlyReduceAvx2(&acc, s, x2Sum)
		}

		e := y[i] - s
		*errorSum += float32(e * e)
		pend.active = false
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			pend.set(xStartIndex, chunk1, hSize, alpha, g == 16)
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
	if pend.active {
		pend.apply(h, x, 0, hSize)
	}
}

// dlyCoreSimple is dlyCoreFused with less bookkeeping: the update of a
// sample is fused into the next sample's dot product only when neither
// sample wraps around the circular buffer and h is a whole number of
// groups, so one update-and-dot kernel call covers all of h. Otherwise the
// pending update runs as its own pass first and the dot product follows the
// C++ chunk loop. The update of every element is the same operation either
// way.
func dlyCoreSimple(g int, xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32) {
	hSize := len(h)
	xSize := len(x)
	var pend dlyPendingAxpy
	pendWhole := false // the pending update covers h with one x stretch at pendStart
	pendStart, pendChunk1 := 0, 0
	var acc [32]float32
	for i := range y {
		var x2Sum, s float32
		chunk1 := min(hSize, xSize-xStartIndex)
		chunk2 := hSize - chunk1
		v1 := chunk1 &^ (g - 1)
		reduced := false
		if v1 == hSize && g != 4 {
			// One stretch, no tails: the Sum kernels reduce in registers.
			xb := x[xStartIndex : xStartIndex+hSize]
			if pend.active && pendWhole {
				xa := x[pendStart : pendStart+hSize]
				if g == 8 {
					s, x2Sum = dlyAxpyDotSse2Sum(h, xa, xb, pend.alpha)
				} else {
					s, x2Sum = dlyAxpyFmaDotAvx2Sum(h, xa, xb, pend.alpha)
				}
			} else {
				if pend.active {
					pend.set(pendStart, pendChunk1, hSize, pend.alpha, g == 16)
					pend.apply(h, x, 0, hSize)
				}
				if g == 8 {
					s, x2Sum = dlyDotSse2Sum(h, xb)
				} else {
					s, x2Sum = dlyDotAvx2Sum(h, xb)
				}
			}
			reduced = true
		} else if pend.active && pendWhole && v1 == hSize {
			dlyZeroAcc(&acc)
			xa := x[pendStart : pendStart+hSize]
			xb := x[xStartIndex : xStartIndex+hSize]
			dlyAxpyDotNeon(h, xa, xb, pend.alpha, (*[8]float32)(acc[:8]))
		} else {
			dlyZeroAcc(&acc)
			if pend.active {
				pend.set(pendStart, pendChunk1, hSize, pend.alpha, g == 16)
				pend.apply(h, x, 0, hSize)
			}
			dlyLaneDot(g, h[:v1], x[xStartIndex:xStartIndex+v1], &acc)
			for k := v1; k < chunk1; k++ {
				xk := x[xStartIndex+k]
				x2Sum += float32(xk * xk)
				s += float32(h[k] * xk)
			}
			v2 := chunk2 &^ (g - 1)
			dlyLaneDot(g, h[chunk1:chunk1+v2], x[:v2], &acc)
			for k := chunk1 + v2; k < hSize; k++ {
				xk := x[k-chunk1]
				x2Sum += float32(xk * xk)
				s += float32(h[k] * xk)
			}
		}
		switch {
		case reduced:
		case g == 4:
			s, x2Sum = dlyReduceNeon((*[8]float32)(acc[:8]), s, x2Sum)
		case g == 8:
			s, x2Sum = dlyReduceSse2((*[16]float32)(acc[:16]), s, x2Sum)
		default:
			s, x2Sum = dlyReduceAvx2(&acc, s, x2Sum)
		}

		e := y[i] - s
		*errorSum += float32(e * e)
		pend.active = false
		if x2Sum > x2SumThreshold && !dlySaturated(y[i]) {
			alpha := float32(smoothing*e) / x2Sum
			pend.active = true
			pend.alpha = alpha
			pendWhole = chunk1 == hSize
			pendStart, pendChunk1 = xStartIndex, chunk1
			*filtersUpdated = true
		}
		if xStartIndex > 0 {
			xStartIndex--
		} else {
			xStartIndex = xSize - 1
		}
	}
	if pend.active {
		pend.set(pendStart, pendChunk1, hSize, pend.alpha, g == 16)
		pend.apply(h, x, 0, hSize)
	}
}

// MatchedFilterCore mirrors the scalar aec3::MatchedFilterCore.
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

// MaxSquarePeakIndex finds the largest peak of squared values in h.
func MaxSquarePeakIndex(h []float32) int { return dlyMaxSquarePeakIndex(h) }

// dlyMaxSquarePeakIndexRef is the scalar MaxSquarePeakIndex of the C++: it
// tracks the even and odd indices separately with strict comparisons, so
// each keeps its earliest maximum.
func dlyMaxSquarePeakIndexRef(h []float32) int {
	if len(h) < 2 {
		return 0
	}
	maxElement1 := h[0] * h[0]
	maxElement2 := h[1] * h[1]
	lagEstimate1 := 0
	lagEstimate2 := 1
	lastIndex := len(h) - 1
	// Keeping track of even & odd max elements separately typically allows the
	// compiler to produce more efficient code.
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
	// In case of odd h size, we have not yet checked the last element.
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

// MatchedFilter produces recursively updated cross-correlation estimates for
// several signal shifts where the intra-shift spacing is uniform
// (matched_filter.h).
type MatchedFilter struct {
	optimization                  Aec3Optimization
	subBlockSize                  int
	filterIntraLagShift           int
	filters                       [][]float32
	accumulatedError              [][]float32
	instantaneousAccumulatedError []float32
	scratchMemory                 []float32
	reportedLagEstimate           MatchedFilterLagEstimate
	hasReportedLagEstimate        bool
	winnerLag                     int
	hasWinnerLag                  bool
	lastDetectedBestLagFilter     int
	numberPreEchoUpdates          int
	excitationLimit               float32
	smoothingFast                 float32
	smoothingSlow                 float32
	matchingFilterThreshold       float32
	detectPreEcho                 bool

	// Quiet-render skip (see renderIsQuiet): quietSkip enables it,
	// quietLimitBits is the bit pattern of the largest quiet magnitude and
	// loudHint the index of a render sample that last exceeded it.
	quietSkip      bool
	quietLimitBits uint32
	loudHint       int
}

// NewMatchedFilter mirrors the MatchedFilter constructor (without the data
// dumper).
func NewMatchedFilter(optimization Aec3Optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters,
	alignmentShiftSubBlocks int, excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold float32,
	detectPreEcho bool) *MatchedFilter {
	m := &MatchedFilter{}
	m.init(optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters, alignmentShiftSubBlocks,
		excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold, detectPreEcho)
	return m
}

func (m *MatchedFilter) init(optimization Aec3Optimization, subBlockSize, windowSizeSubBlocks, numMatchedFilters,
	alignmentShiftSubBlocks int, excitationLimit, smoothingFast, smoothingSlow, matchingFilterThreshold float32,
	detectPreEcho bool) {
	if windowSizeSubBlocks <= 0 || BlockSize%subBlockSize != 0 || subBlockSize%4 != 0 {
		panic("aec3: invalid MatchedFilter dimensions")
	}
	*m = MatchedFilter{
		optimization:              optimization,
		subBlockSize:              subBlockSize,
		filterIntraLagShift:       alignmentShiftSubBlocks * subBlockSize,
		filters:                   make([][]float32, numMatchedFilters),
		lastDetectedBestLagFilter: -1,
		excitationLimit:           excitationLimit,
		smoothingFast:             smoothingFast,
		smoothingSlow:             smoothingSlow,
		matchingFilterThreshold:   matchingFilterThreshold,
		detectPreEcho:             detectPreEcho,
	}
	filterSize := windowSizeSubBlocks * subBlockSize
	for n := range m.filters {
		m.filters[n] = make([]float32, filterSize)
	}
	if detectPreEcho {
		m.accumulatedError = make([][]float32, numMatchedFilters)
		for n := range m.accumulatedError {
			e := make([]float32, filterSize/dlyAccumulatedErrorSubSampleRate)
			for k := range e {
				e[k] = 1
			}
			m.accumulatedError[n] = e
		}
		m.instantaneousAccumulatedError = make([]float32, filterSize/dlyAccumulatedErrorSubSampleRate)
		m.scratchMemory = make([]float32, filterSize)
	}
	var quietLimit float32
	quietLimit, m.quietSkip = MatchedFilterQuietLimit(filterSize, excitationLimit)
	m.quietLimitBits = math.Float32bits(quietLimit)
}

// Quiet-render skip.
//
// A filter core changes state only through its filter update, which a
// capture sample triggers when x2_sum > x2_sum_threshold (and the sample is
// not saturated). When no filter updates during a block, Update has no
// winner: it clears winner_lag and reported_lag_estimate and leaves the
// filters, accumulated_error, last_detected_best_lag_filter and
// number_pre_echo_updates as they are. The error sums and peak indices of
// such a block are computed and discarded. Update therefore returns right
// after clearing the two estimates when it can tell beforehand that
// x2_sum <= x2_sum_threshold holds for every sample of every filter.
//
// instantaneous_accumulated_error and scratch_memory are scratch: the
// accumulated-error core of filter last_detected_best_lag_filter clears and
// rewrites the former in every call that runs the cores, and Update reads it
// only later in the same call, when that filter is the winner; the latter is
// written and read inside one core iteration. A skipped block leaves both
// stale, which no later block can observe.
//
// Bound on x2_sum. Let N be the filter length and suppose |x[k]| <= m for
// every element of the render buffer, with m finite. In every lane x2_sum
// is the value of an expression over the N samples of the filter's window,
// each used once, built from rounded products R(x*x), rounded sums R(a+b)
// and (AVX2 lane) fused R(x*x+c), from accumulators that start at +0; lanes
// differ in the shape of the expression only. With u = 2^-24 and
// eta = 2^-150 (half the smallest subnormal), rounding to nearest gives
// R(t) <= t*(1+u) + eta for a product or fused term t >= 0 and
// R(a+b) <= (a+b)*(1+u) for a sum of floats (sums in the subnormal range are
// exact). By induction over the expression, a node over n samples whose
// longest path has d rounded operations is at most
// (sum of its x[k]^2 + n*eta) * (1+u)^d. No path is longer than N+40: one
// product and at most N-1 chained additions, the additions to the +0
// accumulators, and the lane reduction (at most 5 levels). Hence
//
//	x2_sum <= (N*m^2 + N*eta) * (1+u)^(N+40).
//
// dlyQuietLimit picks m with N*m^2*(1+margin) <= x2_sum_threshold for
// margin = 4*(N+40)*u (dlyQuietMargin), twice the 2*(N+40)*u that bounds
// (1+u)^(N+40) - 1 for N <= 2^16; the spare half covers N*eta, which is
// below x2_sum_threshold*2^-66 for the thresholds (>= 2^-60) that get a
// nonzero m. Smaller thresholds get m = 0: a buffer of zeros gives
// x2_sum = 0 exactly, which never exceeds a threshold >= 0. The skip is
// disabled when the threshold is negative, NaN or infinite or N > 2^16.
//
// The test |x[k]| <= m runs on the bit patterns: for finite m, clearing the
// sign bit of x[k] and comparing as integers orders magnitudes, and puts
// every infinity and NaN above m, so non-finite render samples never count
// as quiet.

// dlyQuietMargin is the relative margin of the quiet limit for a filter of
// n samples.
func dlyQuietMargin(n int) float64 { return 4 * float64(n+40) / (1 << 24) }

// dlyQuietLimit returns the largest magnitude m with
// n*m^2*(1+margin) <= x2SumThreshold, rounded down, and whether the skip
// applies at all.
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
	// One more float32 step down covers the rounding of the float64
	// evaluation above.
	return math.Nextafter32(m, 0), true
}

// MatchedFilterQuietLimit returns the quiet limit of a MatchedFilter with
// filters of filterSize samples and the given excitation limit: when no
// render buffer sample exceeds it in magnitude, no filter updates. It
// reports false when the skip does not apply.
func MatchedFilterQuietLimit(filterSize int, excitationLimit float32) (float32, bool) {
	x2SumThreshold := float32(float32(filterSize)*excitationLimit) * excitationLimit
	return dlyQuietLimit(filterSize, x2SumThreshold, dlyQuietMargin(filterSize))
}

// dlyQuietSkips counts the blocks every MatchedFilter skipped.
var dlyQuietSkips atomic.Uint64

// MatchedFilterQuietSkipsForTesting returns the number of Update calls, over
// all matched filters of the process, that took the quiet-render skip. Tests
// use it to check that a scenario reaches the skip.
func MatchedFilterQuietSkipsForTesting() uint64 { return dlyQuietSkips.Load() }

// renderIsQuiet reports whether every sample of the render buffer has a
// magnitude of at most the quiet limit.
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

// Reset resets the matched filter.
func (m *MatchedFilter) Reset(fullReset bool) {
	for _, f := range m.filters {
		clear(f)
	}
	m.hasWinnerLag = false
	m.hasReportedLagEstimate = false
	if fullReset {
		for _, e := range m.accumulatedError {
			for k := range e {
				e[k] = 1
			}
		}
		m.numberPreEchoUpdates = 0
	}
}

// GetBestLagEstimate returns the current lag estimate, if any.
func (m *MatchedFilter) GetBestLagEstimate() (MatchedFilterLagEstimate, bool) {
	if !m.hasReportedLagEstimate {
		return MatchedFilterLagEstimate{}, false
	}
	return m.reportedLagEstimate, true
}

// GetMaxFilterLag returns the maximum filter lag.
func (m *MatchedFilter) GetMaxFilterLag() int {
	return len(m.filters)*m.filterIntraLagShift + len(m.filters[0])
}

func (m *MatchedFilter) runCore(xStartIndex int, x2SumThreshold, smoothing float32, x, y, h []float32,
	filtersUpdated *bool, errorSum *float32, computePreEcho bool) {
	switch m.optimization {
	case OptSse2:
		MatchedFilterCoreSse2(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError, m.scratchMemory)
	case OptAvx2:
		MatchedFilterCoreAvx2(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError, m.scratchMemory)
	case OptNeon:
		MatchedFilterCoreNeon(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError, m.scratchMemory)
	default:
		MatchedFilterCore(xStartIndex, x2SumThreshold, smoothing, x, y, h, filtersUpdated, errorSum,
			computePreEcho, m.instantaneousAccumulatedError)
	}
}

// Update updates the correlation with the values in the capture buffer.
func (m *MatchedFilter) Update(renderBuffer *DownsampledRenderBuffer, capture []float32, useSlowSmoothing bool) {
	y := capture[:m.subBlockSize]

	smoothing := m.smoothingFast
	if useSlowSmoothing {
		smoothing = m.smoothingSlow
	}

	x2SumThreshold := float32(float32(len(m.filters[0]))*m.excitationLimit) * m.excitationLimit

	// Compute anchor for the matched filter error.
	var errorSumAnchor float32
	for _, v := range y {
		errorSumAnchor += float32(v * v)
	}

	// Apply all matched filters.
	winnerErrorSum := errorSumAnchor
	m.hasWinnerLag = false
	m.hasReportedLagEstimate = false
	if m.quietSkip && m.renderIsQuiet(renderBuffer.Buffer) {
		// No filter can update: there is no winner (see the quiet-render
		// skip above).
		dlyQuietSkips.Add(1)
		return
	}
	alignmentShift := 0
	previousLagEstimate, hasPreviousLagEstimate := 0, false
	winnerIndex := -1
	for n := range m.filters {
		var errorSum float32
		filtersUpdated := false
		computePreEcho := m.detectPreEcho && n == m.lastDetectedBestLagFilter

		xStartIndex := (renderBuffer.Read + alignmentShift + m.subBlockSize - 1) % len(renderBuffer.Buffer)

		m.runCore(xStartIndex, x2SumThreshold, smoothing, renderBuffer.Buffer, y, m.filters[n],
			&filtersUpdated, &errorSum, computePreEcho)

		// Estimate the lag in the matched filter as the distance to the portion
		// in the filter that contributes the most to the matched filter output.
		// This is detected as the peak of the matched filter.
		lagEstimate := MaxSquarePeakIndex(m.filters[n])
		reliable := lagEstimate > 2 && lagEstimate < len(m.filters[n])-10 &&
			errorSum < float32(m.matchingFilterThreshold*errorSumAnchor)

		// Find the best estimate.
		lag := lagEstimate + alignmentShift
		if filtersUpdated && reliable && errorSum < winnerErrorSum {
			winnerErrorSum = errorSum
			winnerIndex = n
			// In case that 2 matched filters return the same winner candidate
			// (overlap region), the one with the smaller index is chosen in
			// order to search for pre-echoes.
			if hasPreviousLagEstimate && previousLagEstimate == lag {
				m.winnerLag = previousLagEstimate
				winnerIndex = n - 1
			} else {
				m.winnerLag = lag
			}
			m.hasWinnerLag = true
		}
		previousLagEstimate, hasPreviousLagEstimate = lag, true
		alignmentShift += m.filterIntraLagShift
	}

	if winnerIndex != -1 {
		m.reportedLagEstimate = MatchedFilterLagEstimate{Lag: m.winnerLag, PreEchoLag: m.winnerLag}
		m.hasReportedLagEstimate = true
		if m.detectPreEcho && m.lastDetectedBestLagFilter == winnerIndex {
			const energyThreshold = 1.0
			if errorSumAnchor > energyThreshold {
				dlyUpdateAccumulatedError(m.instantaneousAccumulatedError, m.accumulatedError[winnerIndex],
					1/errorSumAnchor)
				m.numberPreEchoUpdates++
			}
			if m.numberPreEchoUpdates >= 50 {
				m.reportedLagEstimate.PreEchoLag = dlyComputePreEchoLag(m.accumulatedError[winnerIndex],
					m.winnerLag, winnerIndex*m.filterIntraLagShift)
			} else {
				m.reportedLagEstimate.PreEchoLag = m.winnerLag
			}
		}
		m.lastDetectedBestLagFilter = winnerIndex
	}
}
