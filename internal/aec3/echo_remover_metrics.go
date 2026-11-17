package aec3

import "math"

// Port of echo_remover_metrics.{h,cc}. Histogram reporting is not ported; the
// collection and reset schedule is.

// DbMetric mirrors EchoRemoverMetrics::DbMetric.
type DbMetric struct {
	SumValue   float32
	FloorValue float32
	CeilValue  float32
}

// Update accumulates value into the metric.
func (m *DbMetric) Update(value float32) {
	m.SumValue += value
	m.FloorValue = min(m.FloorValue, value)
	m.CeilValue = max(m.CeilValue, value)
}

// UpdateInstant replaces the sum with value and updates floor and ceiling.
func (m *DbMetric) UpdateInstant(value float32) {
	m.SumValue = value
	m.FloorValue = min(m.FloorValue, value)
	m.CeilValue = max(m.CeilValue, value)
}

// EchoRemoverMetrics handles the metrics collection for the echo remover.
type EchoRemoverMetrics struct {
	blockCounter     int
	erlTimeDomain    DbMetric
	erleTimeDomain   DbMetric
	saturatedCapture bool
	metricsReported  bool
}

// NewEchoRemoverMetrics returns metrics in their reset state.
func NewEchoRemoverMetrics() *EchoRemoverMetrics {
	m := &EchoRemoverMetrics{}
	m.resetMetrics()
	return m
}

func (m *EchoRemoverMetrics) resetMetrics() {
	m.erlTimeDomain = DbMetric{0, 10000, 0}
	m.erleTimeDomain = DbMetric{0, 0, 1000}
	m.saturatedCapture = false
}

// Update updates the metrics with new data. The comfort noise spectrum and
// the suppressor gain are unused, as in the C++.
func (m *EchoRemoverMetrics) Update(aecState *AecState, comfortNoiseSpectrum, suppressorGain *[FftLengthBy2Plus1]float32) {
	m.metricsReported = false
	m.blockCounter++
	if m.blockCounter <= MetricsCollectionBlocks {
		m.erlTimeDomain.UpdateInstant(aecState.ErlTimeDomain())
		m.erleTimeDomain.UpdateInstant(aecState.FullBandErleLog2())
		m.saturatedCapture = m.saturatedCapture || aecState.SaturatedCapture()
		return
	}
	// The C++ reports histograms over the three blocks after collection;
	// only the schedule and the reset are reproduced.
	if m.blockCounter == MetricsCollectionBlocks+3 {
		m.metricsReported = true
		m.blockCounter = 0
		m.resetMetrics()
	}
}

// MetricsReported returns true if the metrics have just been reported.
func (m *EchoRemoverMetrics) MetricsReported() bool { return m.metricsReported }

// UpdateDbMetric updates a banded metric with the averages of the two halves
// of value.
func UpdateDbMetric(value *[FftLengthBy2Plus1]float32, statistic *[2]DbMetric) {
	const numBands = 2
	const bandWidth = 65 / numBands
	const oneByBandWidth = float32(1) / bandWidth
	for k := range statistic {
		var acc float32
		for _, v := range value[bandWidth*k : bandWidth*(k+1)] {
			acc += v
		}
		statistic[k].Update(acc * oneByBandWidth)
	}
}

// TransformDbMetricForReporting transforms a DbMetric value from the linear
// domain into the logarithmic domain.
func TransformDbMetricForReporting(negate bool, minValue, maxValue, offset, scaling, value float32) int {
	newValue := float32(10*erLog10f(float32(value*scaling)+1e-10)) + offset
	if negate {
		newValue = -newValue
	}
	return int(min(max(newValue, minValue), maxValue))
}

// erLog10f is log10f: the double-precision log10 rounded to float.
func erLog10f(x float32) float32 { return float32(math.Log10(float64(x))) }
