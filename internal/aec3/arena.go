package aec3

// A component that needs pointers (slices whose size depends on the config,
// what more than inlineChannels channels need) is two types: xState, a
// pointer-free struct that its parent's state holds by value, and X, a view
// that embeds *xState next to the slices and the views of its children. The
// methods are those of the view; the fields of the state are promoted into
// it.
//
// A canceller keeps the states in one object the garbage collector does not
// read (Ec3Core) and the slices in a few pointer-free arenas. The views are
// not kept: every entry point binds them on its stack. Each view type has
// one method, bind, that lists its slices and children for the binder:
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
// A view made by NewX stands alone: bind(st, nil) attaches it to a state
// and to nothing else, and init allocates its slices.

// span is where a slice of a view lies in its arena; off is negative for a
// nil slice.
type span struct{ off, n int32 }

// bindMode is what a binder does with the slices and pointers it is shown.
type bindMode uint8

const (
	// bindMeasure records the length of every slice in its span and adds
	// them up per arena.
	bindMeasure bindMode = iota + 1
	// bindMove copies every slice to its span of the arenas and points the
	// view at the copy; pointers go to the extras.
	bindMove
	// bindView points the view at the arenas and the extras.
	bindView
)

// bpArenas hold the slices of a block processor and its components.
type bpArenas struct {
	f32  []float32
	fft  []FftData
	spec [][FftLengthBy2Plus1]float32
	i32  []int32
}

// arenaSizes are the lengths of the arenas.
type arenaSizes struct{ f32, fft, spec, i32 int32 }

// bpExtra holds what a block processor keeps outside its state and its
// arenas: the state of more than inlineChannels channels and the optional
// components. A canceller with up to inlineChannels channels and the
// default components has none.
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

// bpNoExtra is what a view binds to without extras; it is never written.
var bpNoExtra bpExtra

// binder binds the views of a block processor to its state.
type binder struct {
	mode bindMode
	a    bpArenas
	n    arenaSizes
	x    *bpExtra
	// hasExtra is set by bindMove when a pointer it moved is not nil.
	hasExtra bool
}

// f32, fft, spec and i32 bind the slice s of a view, which lies at r in the
// arena of its type. They do nothing for a view that stands alone
// (b == nil).
func (b *binder) f32(s *[]float32, r *span) {
	if b != nil {
		bindSlice(b.mode, s, r, b.a.f32, &b.n.f32)
	}
}

func (b *binder) fft(s *[]FftData, r *span) {
	if b != nil {
		bindSlice(b.mode, s, r, b.a.fft, &b.n.fft)
	}
}

func (b *binder) spec(s *[][FftLengthBy2Plus1]float32, r *span) {
	if b != nil {
		bindSlice(b.mode, s, r, b.a.spec, &b.n.spec)
	}
}

func (b *binder) i32(s *[]int32, r *span) {
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
		*s = arena[r.off : r.off+r.n : r.off+r.n]
	case bindMeasure:
		if *s == nil {
			*r = span{off: -1}
			return
		}
		*r = span{off: *n, n: int32(len(*s))}
		*n += r.n
	case bindMove:
		if r.off < 0 {
			return
		}
		d := arena[r.off : r.off+r.n : r.off+r.n]
		copy(d, *s)
		*s = d
	}
}

// bindPtr binds the pointer (or slice) v of a view, which the extras keep
// in home.
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
