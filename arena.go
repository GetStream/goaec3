package goaec3

import (
	"github.com/GetStream/goaec3/internal/aec3"
	"github.com/GetStream/goaec3/internal/bands"
)

// The arrays of a canceller (samples, filter coefficients, spectra, delay
// histograms, band-splitting filters) have sizes that depend on its format
// and config. For the usual sizes they are the fields of one struct, an
// arena class: one pointer-free heap object, of which the canceller holds
// one pointer. A canceller that fits no class allocates each array
// (ecSliceArena).

// ecArena is the arena of a canceller: the arrays of the AEC3 core and the
// band-splitting filters.
type ecArena interface {
	aec3.Arena
	// splitters returns the band-splitting filters, of the capture channels
	// first.
	splitters() []bands.ChannelSplitter
}

// The array types of the arena classes; s returns the array as a slice.
type (
	ecF21k [21504]float32
	ecF25k [25600]float32
	ecF33k [33792]float32
	ecF41k [41984]float32
	ecF47k [47104]float32
	ecF55k [55296]float32
	ecF61k [61440]float32
	ecF76k [76800]float32
	ecF86k [86016]float32

	ecH0   [0]aec3.FftData
	ecH200 [200]aec3.FftData
	ecH224 [224]aec3.FftData
	ecH448 [448]aec3.FftData

	ecS0   [0][aec3.FftLengthBy2Plus1]float32
	ecS184 [184][aec3.FftLengthBy2Plus1]float32
	ecS200 [200][aec3.FftLengthBy2Plus1]float32
	ecS368 [368][aec3.FftLengthBy2Plus1]float32

	ecI0  [0]int32
	ecI2k [2624]int32

	ecP0 [0]bands.ChannelSplitter
	ecP2 [2]bands.ChannelSplitter
	ecP4 [4]bands.ChannelSplitter
)

func (a *ecF21k) s() []float32 { return a[:] }
func (a *ecF25k) s() []float32 { return a[:] }
func (a *ecF33k) s() []float32 { return a[:] }
func (a *ecF41k) s() []float32 { return a[:] }
func (a *ecF47k) s() []float32 { return a[:] }
func (a *ecF55k) s() []float32 { return a[:] }
func (a *ecF61k) s() []float32 { return a[:] }
func (a *ecF76k) s() []float32 { return a[:] }
func (a *ecF86k) s() []float32 { return a[:] }

func (a *ecH0) s() []aec3.FftData   { return a[:] }
func (a *ecH200) s() []aec3.FftData { return a[:] }
func (a *ecH224) s() []aec3.FftData { return a[:] }
func (a *ecH448) s() []aec3.FftData { return a[:] }

func (a *ecS0) s() [][aec3.FftLengthBy2Plus1]float32   { return a[:] }
func (a *ecS184) s() [][aec3.FftLengthBy2Plus1]float32 { return a[:] }
func (a *ecS200) s() [][aec3.FftLengthBy2Plus1]float32 { return a[:] }
func (a *ecS368) s() [][aec3.FftLengthBy2Plus1]float32 { return a[:] }

func (a *ecI2k) s() []int32 { return a[:] }

func (a *ecP0) s() []bands.ChannelSplitter { return a[:] }
func (a *ecP2) s() []bands.ChannelSplitter { return a[:] }
func (a *ecP4) s() []bands.ChannelSplitter { return a[:] }

// ecArray is the pointer to an array type A with elements E.
type ecArray[A, E any] interface {
	*A
	s() []E
}

// ecArenaOf is an arena class: a struct of the arrays F (samples), H
// (filter coefficients), S (spectra), I (histograms) and P (band-splitting
// filters).
type ecArenaOf[F, H, S, I, P any,
	PF ecArray[F, float32], PH ecArray[H, aec3.FftData], PS ecArray[S, [aec3.FftLengthBy2Plus1]float32],
	PI ecArray[I, int32], PP ecArray[P, bands.ChannelSplitter]] struct {
	f F
	h H
	s S
	i I
	p P
}

func (a *ecArenaOf[F, H, S, I, P, PF, PH, PS, PI, PP]) Views() aec3.ArenaViews {
	return aec3.ArenaViews{F32: PF(&a.f).s(), Fft: PH(&a.h).s(), Spec: PS(&a.s).s(), I32: PI(&a.i).s()}
}

func (a *ecArenaOf[F, H, S, I, P, PF, PH, PS, PI, PP]) Alloc(n aec3.ArenaSizes) aec3.Arena {
	return newEcArena(n, 0)
}

func (a *ecArenaOf[F, H, S, I, P, PF, PH, PS, PI, PP]) splitters() []bands.ChannelSplitter {
	return PP(&a.p).s()
}

// ecClass returns a new arena of the class with the arrays F, H, S, I and P.
func ecClass[F, H, S, I, P any,
	PF ecArray[F, float32], PH ecArray[H, aec3.FftData], PS ecArray[S, [aec3.FftLengthBy2Plus1]float32],
	PI ecArray[I, int32], PP ecArray[P, bands.ChannelSplitter]]() ecArena {
	return new(ecArenaOf[F, H, S, I, P, PF, PH, PS, PI, PP])
}

// ecArenaClasses are the arena classes in the order of their size (a
// canceller takes the first that holds what it needs): what a class holds
// of samples, filter coefficients, spectra, histogram bins and
// band-splitting filters, and how to allocate one. The classes are sized
// for the default configs with up to two channels a side, with some room:
// the canceller's samples and the block processor's arrays in one arena, at
// 16, 32 and 48 kHz, for one and for two capture channels, for a block
// processor on one render channel and on two; and the arrays of a block
// processor alone (the one a canceller sets up when stereo content starts
// goes into an arena of its own).
var ecArenaClasses = []struct {
	f, h, s, i, p int
	alloc         func() ecArena
}{
	{21504, 200, 184, 2624, 0, ecClass[ecF21k, ecH200, ecS184, ecI2k, ecP0]},
	{25600, 224, 200, 2624, 0, ecClass[ecF25k, ecH224, ecS200, ecI2k, ecP0]},
	{33792, 224, 200, 2624, 0, ecClass[ecF33k, ecH224, ecS200, ecI2k, ecP0]},
	{33792, 200, 184, 2624, 2, ecClass[ecF33k, ecH200, ecS184, ecI2k, ecP2]},
	{41984, 224, 200, 2624, 0, ecClass[ecF41k, ecH224, ecS200, ecI2k, ecP0]},
	{47104, 224, 200, 2624, 0, ecClass[ecF47k, ecH224, ecS200, ecI2k, ecP0]},
	{47104, 200, 184, 2624, 2, ecClass[ecF47k, ecH200, ecS184, ecI2k, ecP2]},
	{41984, 224, 200, 2624, 4, ecClass[ecF41k, ecH224, ecS200, ecI2k, ecP4]},
	{55296, 224, 200, 2624, 0, ecClass[ecF55k, ecH224, ecS200, ecI2k, ecP0]},
	{55296, 224, 200, 2624, 4, ecClass[ecF55k, ecH224, ecS200, ecI2k, ecP4]},
	{33792, 448, 368, 2624, 0, ecClass[ecF33k, ecH448, ecS368, ecI2k, ecP0]},
	{41984, 448, 368, 2624, 0, ecClass[ecF41k, ecH448, ecS368, ecI2k, ecP0]},
	{55296, 448, 368, 2624, 0, ecClass[ecF55k, ecH448, ecS368, ecI2k, ecP0]},
	{61440, 448, 368, 2624, 4, ecClass[ecF61k, ecH448, ecS368, ecI2k, ecP4]},
	{76800, 448, 368, 2624, 0, ecClass[ecF76k, ecH448, ecS368, ecI2k, ecP0]},
	{86016, 448, 368, 2624, 4, ecClass[ecF86k, ecH448, ecS368, ecI2k, ecP4]},
}

// ecSliceArena is the arena of a canceller that fits no class.
type ecSliceArena struct {
	aec3.Arena
	p []bands.ChannelSplitter
}

func (a *ecSliceArena) Alloc(n aec3.ArenaSizes) aec3.Arena { return newEcArena(n, 0) }

func (a *ecSliceArena) splitters() []bands.ChannelSplitter { return a.p }

// newEcArena returns an arena that holds at least n and numSplitters
// band-splitting filters: of the first class that does, or of slices.
func newEcArena(n aec3.ArenaSizes, numSplitters int) ecArena {
	for _, c := range ecArenaClasses {
		if n.F32 <= c.f && n.Fft <= c.h && n.Spec <= c.s && n.I32 <= c.i && numSplitters <= c.p {
			return c.alloc()
		}
	}
	a := &ecSliceArena{Arena: aec3.NewSliceArena(n)}
	if numSplitters > 0 {
		a.p = make([]bands.ChannelSplitter, numSplitters)
	}
	return a
}
