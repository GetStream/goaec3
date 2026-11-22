package aec3

// Port of echo_remover_metrics.{h,cc}. The histograms are not ported; the
// schedule of collection and reset is.

// DbMetric mirrors EchoRemoverMetrics::DbMetric.
type DbMetric struct {
	SumValue   float32
	FloorValue float32
	CeilValue  float32
}

// Update adds value to the metric.
func (m *DbMetric) Update(value float32) {
	m.SumValue += value
	m.FloorValue = stdMin(m.FloorValue, value)
	m.CeilValue = stdMax(m.CeilValue, value)
}

// UpdateInstant sets the sum to value and updates the floor and the ceiling.
func (m *DbMetric) UpdateInstant(value float32) {
	m.SumValue = value
	m.FloorValue = stdMin(m.FloorValue, value)
	m.CeilValue = stdMax(m.CeilValue, value)
}

// EchoRemoverMetrics collects the metrics of the echo remover.
type EchoRemoverMetrics struct {
	blockCounter     int
	erlTimeDomain    DbMetric
	erleTimeDomain   DbMetric
	saturatedCapture bool
	metricsReported  bool
}

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

// Update takes the state of one block. It does not use the comfort noise
// spectrum and the suppressor gain, as in the C++.
func (m *EchoRemoverMetrics) Update(aecState *AecState, comfortNoiseSpectrum, suppressorGain *[FftLengthBy2Plus1]float32) {
	m.metricsReported = false
	m.blockCounter++
	if m.blockCounter <= MetricsCollectionBlocks {
		m.erlTimeDomain.UpdateInstant(aecState.ErlTimeDomain())
		m.erleTimeDomain.UpdateInstant(aecState.FullBandErleLog2())
		m.saturatedCapture = m.saturatedCapture || aecState.SaturatedCapture()
		return
	}
	// The C++ reports histograms over the three blocks after collection.
	if m.blockCounter == MetricsCollectionBlocks+3 {
		m.metricsReported = true
		m.blockCounter = 0
		m.resetMetrics()
	}
}

// MetricsReported reports whether the last Update reported the metrics.
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
		statistic[k].Update(float32(acc * oneByBandWidth))
	}
}

// TransformDbMetricForReporting converts the value of a DbMetric from the
// linear to the logarithmic domain.
func TransformDbMetricForReporting(negate bool, minValue, maxValue, offset, scaling, value float32) int {
	newValue := float32(10*Log10f(float32(value*scaling)+1e-10)) + offset
	if negate {
		newValue = -newValue
	}
	return int(stSafeClamp(newValue, minValue, maxValue))
}
