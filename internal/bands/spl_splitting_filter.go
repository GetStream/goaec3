package bands

// Port of common_audio/signal_processing/splitting_filter.c and the
// signal_processing_library helpers it uses. Integer arithmetic wraps the way
// the C code does on two's-complement targets.

// bsMaxBandFrameLength is the maximum number of samples in a low/high-band
// frame (10 ms at 64 kHz).
const bsMaxBandFrameLength = 320

// QMF filter coefficients in Q16.
var (
	bsAllPassFilter1 = [3]uint16{6418, 36982, 57261}
	bsAllPassFilter2 = [3]uint16{21333, 49062, 63010}
)

// SatW32ToW16 mirrors WebRtcSpl_SatW32ToW16.
func SatW32ToW16(value32 int32) int16 {
	out16 := int16(value32)
	if value32 > 32767 {
		out16 = 32767
	} else if value32 < -32768 {
		out16 = -32768
	}
	return out16
}

// SubSatW32 mirrors WebRtcSpl_SubSatW32.
func SubSatW32(a, b int32) int32 {
	diff := int32(uint32(a) - uint32(b))
	// a - b can't overflow if a and b have the same sign. If they have
	// different signs, a - b has the same sign as a iff it didn't overflow.
	if (a < 0) != (b < 0) && (a < 0) != (diff < 0) {
		if diff < 0 {
			return 2147483647
		}
		return -2147483648
	}
	return diff
}

// bsScaleDiff32 mirrors WEBRTC_SPL_SCALEDIFF32(A, B, C):
// C + (B >> 16) * A + (((uint32_t)(B & 0x0000FFFF) * A) >> 16).
// The sum is formed in uint32 (int32 + uint32 promotes) and stored as int32.
func bsScaleDiff32(a uint16, b, c int32) int32 {
	hi := c + (b>>16)*int32(a)
	lo := (uint32(b&0x0000FFFF) * uint32(a)) >> 16
	return int32(uint32(hi) + lo)
}

// bsAllPassQMF mirrors WebRtcSpl_AllPassQMF. inData is overwritten.
func bsAllPassQMF(inData []int32, outData []int32, filterCoefficients *[3]uint16, filterState *[6]int32) {
	n := len(inData)
	outData = outData[:n]

	// First all-pass cascade; filter from inData to outData.
	diff := SubSatW32(inData[0], filterState[1])
	outData[0] = bsScaleDiff32(filterCoefficients[0], diff, filterState[0])
	for k := 1; k < n; k++ {
		diff = SubSatW32(inData[k], outData[k-1])
		outData[k] = bsScaleDiff32(filterCoefficients[0], diff, inData[k-1])
	}
	filterState[0] = inData[n-1]
	filterState[1] = outData[n-1]

	// Second all-pass cascade; filter from outData to inData.
	diff = SubSatW32(outData[0], filterState[3])
	inData[0] = bsScaleDiff32(filterCoefficients[1], diff, filterState[2])
	for k := 1; k < n; k++ {
		diff = SubSatW32(outData[k], inData[k-1])
		inData[k] = bsScaleDiff32(filterCoefficients[1], diff, outData[k-1])
	}
	filterState[2] = outData[n-1]
	filterState[3] = inData[n-1]

	// Third all-pass cascade; filter from inData to outData.
	diff = SubSatW32(inData[0], filterState[5])
	outData[0] = bsScaleDiff32(filterCoefficients[2], diff, filterState[4])
	for k := 1; k < n; k++ {
		diff = SubSatW32(inData[k], outData[k-1])
		outData[k] = bsScaleDiff32(filterCoefficients[2], diff, inData[k-1])
	}
	filterState[4] = inData[n-1]
	filterState[5] = outData[n-1]
}

// AnalysisQMF mirrors WebRtcSpl_AnalysisQMF: it splits inData (even length,
// at most 640 samples) into lowBand and highBand of len(inData)/2 samples.
func AnalysisQMF(inData []int16, lowBand, highBand []int16, filterState1, filterState2 *[6]int32) {
	var halfIn1, halfIn2, filter1, filter2 [bsMaxBandFrameLength]int32
	bandLength := len(inData) / 2

	// Split even and odd samples. Also shift them to Q10.
	for i, k := 0, 0; i < bandLength; i, k = i+1, k+2 {
		halfIn2[i] = int32(inData[k]) * (1 << 10)
		halfIn1[i] = int32(inData[k+1]) * (1 << 10)
	}

	// All pass filter even and odd samples, independently.
	bsAllPassQMF(halfIn1[:bandLength], filter1[:bandLength], &bsAllPassFilter1, filterState1)
	bsAllPassQMF(halfIn2[:bandLength], filter2[:bandLength], &bsAllPassFilter2, filterState2)

	// Take the sum and difference of filtered version of odd and even
	// branches to get upper & lower band.
	lowBand = lowBand[:bandLength]
	highBand = highBand[:bandLength]
	for i := 0; i < bandLength; i++ {
		tmp := (filter1[i] + filter2[i] + 1024) >> 11
		lowBand[i] = SatW32ToW16(tmp)

		tmp = (filter1[i] - filter2[i] + 1024) >> 11
		highBand[i] = SatW32ToW16(tmp)
	}
}

// SynthesisQMF mirrors WebRtcSpl_SynthesisQMF: it merges lowBand and highBand
// (len(lowBand) samples each, at most 320) into 2*len(lowBand) samples.
func SynthesisQMF(lowBand, highBand []int16, outData []int16, filterState1, filterState2 *[6]int32) {
	var halfIn1, halfIn2, filter1, filter2 [bsMaxBandFrameLength]int32
	bandLength := len(lowBand)
	highBand = highBand[:bandLength]

	// Obtain the sum and difference channels out of upper and lower-band
	// channels. Also shift to Q10 domain.
	for i := 0; i < bandLength; i++ {
		tmp := int32(lowBand[i]) + int32(highBand[i])
		halfIn1[i] = tmp * (1 << 10)
		tmp = int32(lowBand[i]) - int32(highBand[i])
		halfIn2[i] = tmp * (1 << 10)
	}

	// All-pass filter the sum and difference channels.
	bsAllPassQMF(halfIn1[:bandLength], filter1[:bandLength], &bsAllPassFilter2, filterState1)
	bsAllPassQMF(halfIn2[:bandLength], filter2[:bandLength], &bsAllPassFilter1, filterState2)

	// The filtered signals are even and odd samples of the output. Combine
	// them. The signals are Q10 should shift them back to Q0 and take care of
	// saturation.
	outData = outData[:2*bandLength]
	for i, k := 0, 0; i < bandLength; i++ {
		tmp := (filter2[i] + 512) >> 10
		outData[k] = SatW32ToW16(tmp)
		k++

		tmp = (filter1[i] + 512) >> 10
		outData[k] = SatW32ToW16(tmp)
		k++
	}
}
