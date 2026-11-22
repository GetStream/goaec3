package aec3

// The scalar lane, several filters in step.
//
// MatchedFilterCore accumulates h*x of a capture sample in one chain, so a
// filter costs the latency of an addition per coefficient. The filters of a
// MatchedFilter share no state, and their chains run side by side:
// dlyCoreScalar runs the sample loop of MatchedFilterCore for up to
// dlyScalarMax filters in step. It computes the same values in another
// arrangement.
//
// x*x and h*x are separate accumulations. The x*x sum of a sample is a chain
// over its window from +0; the chains of eight samples run side by side over
// one table of squares.
//
// The update of a sample is deferred into the h*x accumulation of the next
// one, or a final pass, which updates each element of h right before reading
// it. An element sees the operations of a separate update pass.
//
// The accumulated error is the squared difference of y and the running h*x
// sum after every fourth coefficient. The h*x kernels record the running sum
// of their first filter, and a pass over the record adds the squares.

// dlyScalarMax is the number of filters dlyCoreScalar runs in step: three
// chains fill the latency of an addition.
const dlyScalarMax = 3

// dlyScalarFilter is one filter in a call of dlyCoreScalar.
type dlyScalarFilter struct {
	h        []float32
	xStart   int // start index of the current sample
	errorSum float32
	updated  bool
	// accErr receives the accumulated error, len(h)/4 values that the caller
	// zeroed, or is nil.
	accErr []float32

	// The update of the previous sample, not yet applied.
	alpha   float32
	pending bool

	s  float32           // h*x of the current sample
	x2 [dlyX2Run]float32 // x*x of the samples of the current run

	idle     bool // no sample of the current run updates the filter
	idleRuns int  // runs whose h*x sums were not computed
}

// dlyScalarStream is the work of one filter in a kernel call: the stretch of
// h, the render samples xb it is read against, the render samples xa and the
// alpha of the update that an update kernel applies first, and the running
// h*x sum.
type dlyScalarStream struct {
	h, xa, xb []float32
	alpha, s  float32
}

// dlyScalarScratch returns the scratch dlyCoreScalar needs for filters of
// hSize coefficients: the record of the running sum and the table of
// squares.
func dlyScalarScratch(hSize int) int { return 2*hSize + dlyX2Run }

// dlyScalarX2 stores in out[t], for t < 8, the sum of sq[t:t+n] accumulated
// in order from +0. len(sq) >= n+7.
func dlyScalarX2(sq []float32, n int, out *[8]float32) {
	sq = sq[: n+7 : n+7]
	var q0, q1, q2, q3, q4, q5, q6, q7 float32
	for k := 0; k < len(sq)-7; k++ {
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
	*out = [8]float32{q0, q1, q2, q3, q4, q5, q6, q7}
}

// x2Run stores in f.x2[j], for j < r, the x*x sum of the sample j places
// after the current one: the squares of its window x[(f.xStart-j+k) mod
// len(x)], k < len(f.h), added in order from +0. len(sq) >=
// len(f.h)+dlyX2Run.
func (f *dlyScalarFilter) x2Run(x, sq []float32, r int) {
	hSize, xSize := len(f.h), len(x)
	// sq[i] is the square of the sample i places after the start of the last
	// window. The window of sample j starts at sq[r-1-j].
	lo := f.xStart - (r - 1)
	if lo < 0 {
		lo += xSize
	}
	span := hSize + r - 1
	sq = sq[:hSize+dlyX2Run]
	n1 := min(span, xSize-lo)
	for i, v := range x[lo : lo+n1] {
		sq[i] = float32(v * v)
	}
	for i, v := range x[:span-n1] {
		sq[n1+i] = float32(v * v)
	}
	clear(sq[span:])
	var c [dlyX2Run]float32
	for t := 0; t < r; t += 8 {
		dlyScalarX2(sq[t:], hSize, (*[8]float32)(c[t:t+8]))
	}
	for j := 0; j < r; j++ {
		f.x2[j] = c[r-1-j]
	}
}

// dlyScalarDots computes the h*x sums of the current sample for the filters
// fs[i] with sel[i] set, which all have a pending update (upd) or all have
// none. It records the running sum of the first of them in pre.
func dlyScalarDots(fs []dlyScalarFilter, sel *[dlyScalarMax]bool, upd bool, x, pre []float32) {
	var st [dlyScalarMax]dlyScalarStream
	var idx [dlyScalarMax]int
	n := 0
	for i := range fs {
		if sel[i] {
			idx[n] = i
			st[n].alpha = fs[i].alpha
			n++
		}
	}
	if n == 0 {
		return
	}
	hSize, xSize := len(fs[0].h), len(x)
	for k0 := 0; k0 < hSize; {
		// The stretch up to the next wrap of any window.
		k1 := hSize
		for m := 0; m < n; m++ {
			xb := fs[idx[m]].xStart + k0
			if xb >= xSize {
				xb -= xSize
			}
			k1 = min(k1, k0+xSize-xb)
			if upd {
				// The update reads the window of the previous sample, one
				// place further on.
				xa := xb + 1
				if xa == xSize {
					xa = 0
				}
				k1 = min(k1, k0+xSize-xa)
			}
		}
		for m := 0; m < n; m++ {
			f := &fs[idx[m]]
			xb := f.xStart + k0
			if xb >= xSize {
				xb -= xSize
			}
			st[m].h = f.h[k0:k1]
			st[m].xb = x[xb : xb+k1-k0]
			if upd {
				xa := xb + 1
				if xa == xSize {
					xa = 0
				}
				st[m].xa = x[xa : xa+k1-k0]
			}
		}
		p := pre[k0:k1]
		switch {
		case upd && n == 1:
			dlyScalarAxpyDot1(p, (*[1]dlyScalarStream)(st[:1]))
		case upd && n == 2:
			dlyScalarAxpyDot2(p, (*[2]dlyScalarStream)(st[:2]))
		case upd:
			dlyScalarAxpyDot3(p, &st)
		case n == 1:
			dlyScalarDot1(p, (*[1]dlyScalarStream)(st[:1]))
		case n == 2:
			dlyScalarDot2(p, (*[2]dlyScalarStream)(st[:2]))
		default:
			dlyScalarDot3(p, &st)
		}
		k0 = k1
	}
	for m := 0; m < n; m++ {
		fs[idx[m]].s = st[m].s
	}
}

// dlyCoreScalar runs the sample loop of MatchedFilterCore for the filters fs
// in step. There are at most dlyScalarMax of them, of one length, which is
// at most len(x)-dlyX2Run. Only fs[0] may compute the accumulated error.
// len(scratch) >= dlyScalarScratch(len(fs[0].h)).
//
// With skipIdle, a filter without accumulated error computes neither the h*x
// sums nor the error sum of a run of samples that do not update it. idleRuns
// counts such runs.
func dlyCoreScalar(fs []dlyScalarFilter, x2SumThreshold, smoothing float32, x, y, scratch []float32,
	skipIdle bool) {
	hSize, xSize := len(fs[0].h), len(x)
	pre := scratch[:hSize]
	sq := scratch[hSize : 2*hSize+dlyX2Run]
	for i0 := 0; i0 < len(y); i0 += dlyX2Run {
		r := min(dlyX2Run, len(y)-i0)
		for i := range fs {
			f := &fs[i]
			f.x2Run(x, sq, r)
			// The update of the sample before the run, if any, is applied
			// with the first h*x sum of the run.
			f.idle = skipIdle && f.accErr == nil && !f.pending
			for j := 0; j < r && f.idle; j++ {
				f.idle = !(f.x2[j] > x2SumThreshold && !dlySaturated(y[i0+j]))
			}
			if f.idle {
				f.idleRuns++
			}
		}
		for j := 0; j < r; j++ {
			yi := y[i0+j]
			// The filters with a pending update and those without go to
			// different kernels. The set of fs[0] runs last, so that pre
			// holds its running sum.
			var with, without [dlyScalarMax]bool
			for i := range fs {
				with[i] = fs[i].pending
				without[i] = !fs[i].pending && !fs[i].idle
			}
			if fs[0].pending {
				dlyScalarDots(fs, &without, false, x, pre)
				dlyScalarDots(fs, &with, true, x, pre)
			} else {
				dlyScalarDots(fs, &with, true, x, pre)
				dlyScalarDots(fs, &without, false, x, pre)
			}
			if accErr := fs[0].accErr; accErr != nil {
				for g := range accErr {
					e := yi - pre[4*g+3]
					accErr[g] += float32(e * e)
				}
			}
			for i := range fs {
				f := &fs[i]
				if !f.idle {
					e := yi - f.s
					f.errorSum += float32(e * e)
					f.pending = false
					if x2Sum := f.x2[j]; x2Sum > x2SumThreshold && !dlySaturated(yi) {
						f.alpha = float32(smoothing*e) / x2Sum
						f.pending = true
						f.updated = true
					}
				}
				if f.xStart > 0 {
					f.xStart--
				} else {
					f.xStart = xSize - 1
				}
			}
		}
	}
	// The update still pending after the last sample, on the window of that
	// sample.
	for i := range fs {
		f := &fs[i]
		if !f.pending {
			continue
		}
		xa := f.xStart + 1
		if xa == xSize {
			xa = 0
		}
		chunk1 := min(hSize, xSize-xa)
		dlyAxpyRef(f.h[:chunk1], x[xa:xa+chunk1], f.alpha)
		dlyAxpyRef(f.h[chunk1:], x[:hSize-chunk1], f.alpha)
		f.pending = false
	}
}

// The kernels below run len(pre) coefficients of one, two or three filters.
// The Dot kernels add h*xb to the running sum of each filter; the AxpyDot
// kernels first apply h += alpha*xa, unfused. A kernel stores the running
// sum of its first filter after every coefficient in pre.

func dlyScalarDot1(pre []float32, st *[1]dlyScalarStream) {
	n := len(pre)
	h0, xb0 := st[0].h[:n:n], st[0].xb[:n:n]
	s0 := st[0].s
	for k := range pre {
		s0 += float32(h0[k] * xb0[k])
		pre[k] = s0
	}
	st[0].s = s0
}

func dlyScalarDot2(pre []float32, st *[2]dlyScalarStream) {
	n := len(pre)
	h0, xb0 := st[0].h[:n:n], st[0].xb[:n:n]
	h1, xb1 := st[1].h[:n:n], st[1].xb[:n:n]
	s0, s1 := st[0].s, st[1].s
	for k := range pre {
		s0 += float32(h0[k] * xb0[k])
		pre[k] = s0
		s1 += float32(h1[k] * xb1[k])
	}
	st[0].s, st[1].s = s0, s1
}

func dlyScalarDot3(pre []float32, st *[3]dlyScalarStream) {
	n := len(pre)
	h0, xb0 := st[0].h[:n:n], st[0].xb[:n:n]
	h1, xb1 := st[1].h[:n:n], st[1].xb[:n:n]
	h2, xb2 := st[2].h[:n:n], st[2].xb[:n:n]
	s0, s1, s2 := st[0].s, st[1].s, st[2].s
	for k := range pre {
		s0 += float32(h0[k] * xb0[k])
		pre[k] = s0
		s1 += float32(h1[k] * xb1[k])
		s2 += float32(h2[k] * xb2[k])
	}
	st[0].s, st[1].s, st[2].s = s0, s1, s2
}

func dlyScalarAxpyDot1(pre []float32, st *[1]dlyScalarStream) {
	n := len(pre)
	h0, xb0 := st[0].h[:n:n], st[0].xb[:n:n]
	xa0, al0 := st[0].xa[:n:n], st[0].alpha
	s0 := st[0].s
	for k := range pre {
		v0 := h0[k] + float32(al0*xa0[k])
		h0[k] = v0
		s0 += float32(v0 * xb0[k])
		pre[k] = s0
	}
	st[0].s = s0
}

func dlyScalarAxpyDot2(pre []float32, st *[2]dlyScalarStream) {
	n := len(pre)
	h0, xb0 := st[0].h[:n:n], st[0].xb[:n:n]
	xa0, al0 := st[0].xa[:n:n], st[0].alpha
	h1, xb1 := st[1].h[:n:n], st[1].xb[:n:n]
	xa1, al1 := st[1].xa[:n:n], st[1].alpha
	s0, s1 := st[0].s, st[1].s
	for k := range pre {
		v0 := h0[k] + float32(al0*xa0[k])
		h0[k] = v0
		s0 += float32(v0 * xb0[k])
		pre[k] = s0
		v1 := h1[k] + float32(al1*xa1[k])
		h1[k] = v1
		s1 += float32(v1 * xb1[k])
	}
	st[0].s, st[1].s = s0, s1
}

func dlyScalarAxpyDot3(pre []float32, st *[3]dlyScalarStream) {
	n := len(pre)
	h0, xb0 := st[0].h[:n:n], st[0].xb[:n:n]
	xa0, al0 := st[0].xa[:n:n], st[0].alpha
	h1, xb1 := st[1].h[:n:n], st[1].xb[:n:n]
	xa1, al1 := st[1].xa[:n:n], st[1].alpha
	h2, xb2 := st[2].h[:n:n], st[2].xb[:n:n]
	xa2, al2 := st[2].xa[:n:n], st[2].alpha
	s0, s1, s2 := st[0].s, st[1].s, st[2].s
	for k := range pre {
		v0 := h0[k] + float32(al0*xa0[k])
		h0[k] = v0
		s0 += float32(v0 * xb0[k])
		pre[k] = s0
		v1 := h1[k] + float32(al1*xa1[k])
		h1[k] = v1
		s1 += float32(v1 * xb1[k])
		v2 := h2[k] + float32(al2*xa2[k])
		h2[k] = v2
		s2 += float32(v2 * xb2[k])
	}
	st[0].s, st[1].s, st[2].s = s0, s1, s2
}
