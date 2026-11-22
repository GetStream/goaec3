package aec3

// A component that needs pointers is two types: xState, a pointer-free
// struct that the state of its parent holds by value, and X, a view that
// embeds *xState next to the slices and the views of its children. The
// methods are those of the view.
//
// A canceller keeps the states in one object (Ec3Core) and the slices in
// pointer-free arenas. Every entry point binds the views on its stack. A
// view type lists its slices and children for the binder in its bind method:
//
//	func (x *X) bind(st *xState, b *binder) {
//		x.xState = st
//		b.f32(&x.samples, &st.samplesSpan)
//		if b != nil {
//			bindPtr(b, &x.more, &b.x.xMore)
//		}
//		x.child.bind(&st.child, b)
//	}
//
// A view made by NewX stands alone: bind(st, bindAlone) attaches it to its
// state only, and init allocates its slices.

// span is the place of a slice of a view in its arena. off is negative for a
// nil slice.
type span struct{ off, end int }

// bindMode is what a binder does with the slices and pointers it is shown.
type bindMode uint8

const (
	// bindMeasure records the length of every slice in its span and adds
	// them up per arena.
	bindMeasure bindMode = iota + 1
	// bindMove copies every slice to its span of the arenas and points the
	// view at the copy. Pointers go to the extras.
	bindMove
	// bindView points the view at the arenas and the extras.
	bindView
)

// bpArenas holds the slices of a block processor and its components.
type bpArenas struct {
	f32  []float32
	fft  []FftData
	spec [][FftLengthBy2Plus1]float32
	i32  []int32
}

// arenaSizes holds the lengths of the arenas.
type arenaSizes struct{ f32, fft, spec, i32 int32 }

// bpExtra holds what a block processor keeps outside its state and its
// arenas: the state of more than inlineChannels channels and the optional
// components.
type bpExtra struct {
	erMore  *erMore
	subMore *subMore

	aecMore                      *aecMore
	soaMore                      *soaMore
	faMore                       *faChannels
	rmMore                       *rmChannels
	fbErleMore                   *fbErleChannels
	sbErleMore                   *sbErleChannels
	signalDependentErleEstimator *SignalDependentErleEstimator
	sgSubbandNearendDetector     *SubbandNearendDetector
	sgMore                       *sgMore
	dominantNearendMore          *dominantNearendMore
	cngMore                      *cngMore
	sfMore                       *sfMore
}

// bpNoExtra is what a view without extras binds to. It is never written.
var bpNoExtra bpExtra

// bindAlone is the binder of a view that stands alone. It is never written.
var bindAlone = &binder{notView: -1, x: &bpNoExtra}

// binder binds the views of a block processor to its state.
type binder struct {
	mode bindMode
	// notView is 0 with bindView and -1 otherwise: r.off|notView >= 0 holds
	// for a slice to bind for a call, and for no other.
	notView int
	a       bpArenas
	n       arenaSizes
	x       *bpExtra
	// hasExtra is set by bindMove when it moves a pointer that is not nil.
	hasExtra bool
}

// f32, fft, spec and i32 bind the slice s of a view, which lies at r in the
// arena of its type. They do nothing for a view that stands alone. The
// binding for a call inlines; the rest is in the Slow methods.
func (b *binder) f32(s *[]float32, r *span) {
	if r.off|b.notView < 0 {
		b.f32Slow(s, r)
		return
	}
	*s = b.a.f32[r.off:r.end:r.end]
}

//go:noinline
func (b *binder) f32Slow(s *[]float32, r *span) {
	if b != nil {
		bindSlice(b.mode, s, r, b.a.f32, &b.n.f32)
	}
}

func (b *binder) fft(s *[]FftData, r *span) {
	if r.off|b.notView < 0 {
		b.fftSlow(s, r)
		return
	}
	*s = b.a.fft[r.off:r.end:r.end]
}

//go:noinline
func (b *binder) fftSlow(s *[]FftData, r *span) {
	if b != nil {
		bindSlice(b.mode, s, r, b.a.fft, &b.n.fft)
	}
}

func (b *binder) spec(s *[][FftLengthBy2Plus1]float32, r *span) {
	if r.off|b.notView < 0 {
		b.specSlow(s, r)
		return
	}
	*s = b.a.spec[r.off:r.end:r.end]
}

//go:noinline
func (b *binder) specSlow(s *[][FftLengthBy2Plus1]float32, r *span) {
	if b != nil {
		bindSlice(b.mode, s, r, b.a.spec, &b.n.spec)
	}
}

func (b *binder) i32(s *[]int32, r *span) {
	if r.off|b.notView < 0 {
		b.i32Slow(s, r)
		return
	}
	*s = b.a.i32[r.off:r.end:r.end]
}

//go:noinline
func (b *binder) i32Slow(s *[]int32, r *span) {
	if b != nil {
		bindSlice(b.mode, s, r, b.a.i32, &b.n.i32)
	}
}

func bindSlice[T any](mode bindMode, s *[]T, r *span, arena []T, n *int32) {
	switch mode {
	case bindView:
		if r.off < 0 {
			*s = nil
			return
		}
		*s = arena[r.off:r.end:r.end]
	case bindMeasure:
		if *s == nil {
			*r = span{off: -1}
			return
		}
		*r = span{off: int(*n), end: int(*n) + len(*s)}
		*n = int32(r.end)
	case bindMove:
		if r.off < 0 {
			return
		}
		d := arena[r.off:r.end:r.end]
		copy(d, *s)
		*s = d
	}
}

// bindPtr binds the pointer or slice v of a view, which the extras keep in
// home.
func bindPtr[T comparable](b *binder, v *T, home *T) {
	switch b.mode {
	case bindView:
		*v = *home
	case bindMove:
		var zero T
		*home = *v
		if *v != zero {
			b.hasExtra = true
		}
	}
}
