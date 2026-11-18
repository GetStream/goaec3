package bands

// Filter bank layout, as in three_band_filter_bank.h.
const (
	Sparsity       = 4
	StrideLog2     = 2
	Stride         = 1 << StrideLog2
	NumZeroFilters = 2
	FilterSize     = 4
	MemorySize     = FilterSize*Stride - 1
)

// ThreeBandFilterBank class constants.
const (
	NumBands          = 3
	FullBandSize      = 480
	SplitBandSize     = FullBandSize / NumBands
	NumNonZeroFilters = Sparsity*NumBands - NumZeroFilters
)

const (
	bsSubSampling      = NumBands
	bsDctSize          = NumBands
	bsZeroFilterIndex1 = 3
	bsZeroFilterIndex2 = 9
)

// The Matlab code to generate these coefficients is:
//
//	N = kNumBands * kSparsity * kFilterSize - 1;
//	h = fir1(N, 1 / (2 * kNumBands), kaiser(N + 1, 3.5));
//	reshape(h, kNumBands * kSparsity, kFilterSize);
var bsFilterCoeffs = [NumNonZeroFilters][FilterSize]float32{
	{-0.00047749, -0.00496888, +0.16547118, +0.00425496},
	{-0.00173287, -0.01585778, +0.14989004, +0.00994113},
	{-0.00304815, -0.02536082, +0.12154542, +0.01157993},
	{-0.00346946, -0.02587886, +0.04760441, +0.00607594},
	{-0.00154717, -0.01136076, +0.01387458, +0.00186353},
	{+0.00186353, +0.01387458, -0.01136076, -0.00154717},
	{+0.00607594, +0.04760441, -0.02587886, -0.00346946},
	{+0.00983212, +0.08543175, -0.02982767, -0.00383509},
	{+0.00994113, +0.14989004, -0.01585778, -0.00173287},
	{+0.00425496, +0.16547118, -0.00496888, -0.00047749},
}

// bsSqrt3 is std::numbers::sqrt3_v<float>.
const bsSqrt3 float32 = 1.7320508075688772935274463415058723669428

var bsDctModulation = [NumNonZeroFilters][bsDctSize]float32{
	{2, 2, 2},
	{bsSqrt3, 0, -bsSqrt3},
	{1, -2, 1},
	{-1, 2, -1},
	{-bsSqrt3, 0, bsSqrt3},
	{-2, -2, -2},
	{-bsSqrt3, 0, bsSqrt3},
	{-1, 2, -1},
	{1, -2, 1},
	{bsSqrt3, 0, -bsSqrt3},
}

// bsFilterCore filters in with filter using a shift by inShift, taking into
// account the previous state.
func bsFilterCore(filter *[FilterSize]float32, in *[SplitBandSize]float32, inShift int,
	out *[SplitBandSize]float32, state *[MemorySize]float32) {
	*out = [SplitBandSize]float32{}

	for k := 0; k < inShift; k++ {
		for i, j := 0, MemorySize+k-inShift; i < FilterSize; i, j = i+1, j-Stride {
			out[k] += float32(state[j] * filter[i])
		}
	}

	for k, shift := inShift, 0; k < FilterSize*Stride; k, shift = k+1, shift+1 {
		loopLimit := min(FilterSize, 1+(shift>>StrideLog2))
		for i, j := 0, shift; i < loopLimit; i, j = i+1, j-Stride {
			out[k] += float32(in[j] * filter[i])
		}
		for i, j := loopLimit, MemorySize+shift-loopLimit*Stride; i < FilterSize; i, j = i+1, j-Stride {
			out[k] += float32(state[j] * filter[i])
		}
	}

	for k, shift := FilterSize*Stride, FilterSize*Stride-inShift; k < SplitBandSize; k, shift = k+1, shift+1 {
		for i, j := 0, shift; i < FilterSize; i, j = i+1, j-Stride {
			out[k] += float32(in[j] * filter[i])
		}
	}

	// Update current state.
	copy(state[:], in[SplitBandSize-MemorySize:])
}

func bsFilterIndex(index int) int {
	if index < bsZeroFilterIndex1 {
		return index
	}
	if index < bsZeroFilterIndex2 {
		return index - 1
	}
	return index - 2
}

// threeBandFilterBankRef is the scalar three_band_filter_bank.cc, operation
// for operation. It is the spec the archsimd ThreeBandFilterBank kernels are
// tested against and the implementation used without them.
type threeBandFilterBankRef struct {
	stateAnalysis  [NumNonZeroFilters][MemorySize]float32
	stateSynthesis [NumNonZeroFilters][MemorySize]float32

	// Scratch.
	inSubsampled  [SplitBandSize]float32
	outSubsampled [SplitBandSize]float32
}

func NewThreeBandFilterBank() *ThreeBandFilterBank { return &ThreeBandFilterBank{} }

// analysis is ThreeBandFilterBank::Analysis.
func (f *threeBandFilterBankRef) analysis(in *[FullBandSize]float32, out *[NumBands][]float32) {
	var outBands [NumBands]*[SplitBandSize]float32
	for band := range outBands {
		outBands[band] = (*[SplitBandSize]float32)(out[band])
		*outBands[band] = [SplitBandSize]float32{}
	}

	inSubsampled := &f.inSubsampled
	outSubsampled := &f.outSubsampled
	for downsamplingIndex := 0; downsamplingIndex < bsSubSampling; downsamplingIndex++ {
		// Downsample to form the filter input.
		for k := 0; k < SplitBandSize; k++ {
			inSubsampled[k] = in[(bsSubSampling-1)-downsamplingIndex+bsSubSampling*k]
		}

		for inShift := 0; inShift < Stride; inShift++ {
			// Choose filter, skip zero filters.
			index := downsamplingIndex + inShift*bsSubSampling
			if index == bsZeroFilterIndex1 || index == bsZeroFilterIndex2 {
				continue
			}
			filterIndex := bsFilterIndex(index)
			dctModulation := &bsDctModulation[filterIndex]

			// Filter.
			bsFilterCore(&bsFilterCoeffs[filterIndex], inSubsampled, inShift, outSubsampled,
				&f.stateAnalysis[filterIndex])

			// Band and modulate the output.
			for band := 0; band < NumBands; band++ {
				outBand := outBands[band]
				m := dctModulation[band]
				for n := 0; n < SplitBandSize; n++ {
					outBand[n] += float32(m * outSubsampled[n])
				}
			}
		}
	}
}

// synthesis is ThreeBandFilterBank::Synthesis.
func (f *threeBandFilterBankRef) synthesis(in *[NumBands][]float32, out *[FullBandSize]float32) {
	var inBands [NumBands]*[SplitBandSize]float32
	for band := range inBands {
		inBands[band] = (*[SplitBandSize]float32)(in[band])
	}

	*out = [FullBandSize]float32{}
	inSubsampled := &f.inSubsampled
	outSubsampled := &f.outSubsampled
	for upsamplingIndex := 0; upsamplingIndex < bsSubSampling; upsamplingIndex++ {
		for inShift := 0; inShift < Stride; inShift++ {
			// Choose filter, skip zero filters.
			index := upsamplingIndex + inShift*bsSubSampling
			if index == bsZeroFilterIndex1 || index == bsZeroFilterIndex2 {
				continue
			}
			filterIndex := bsFilterIndex(index)
			dctModulation := &bsDctModulation[filterIndex]

			// Prepare filter input by modulating the banded input.
			*inSubsampled = [SplitBandSize]float32{}
			for band := 0; band < NumBands; band++ {
				inBand := inBands[band]
				m := dctModulation[band]
				for n := 0; n < SplitBandSize; n++ {
					inSubsampled[n] += float32(m * inBand[n])
				}
			}

			// Filter.
			bsFilterCore(&bsFilterCoeffs[filterIndex], inSubsampled, inShift, outSubsampled,
				&f.stateSynthesis[filterIndex])

			// Upsample.
			const upsamplingScaling float32 = bsSubSampling
			for k := 0; k < SplitBandSize; k++ {
				out[upsamplingIndex+bsSubSampling*k] += float32(upsamplingScaling * outSubsampled[k])
			}
		}
	}
}
