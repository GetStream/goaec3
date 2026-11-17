package aec3

// ClockdriftDetectorLevel mirrors ClockdriftDetector::Level.
type ClockdriftDetectorLevel int

const (
	ClockdriftLevelNone ClockdriftDetectorLevel = iota
	ClockdriftLevelProbable
	ClockdriftLevelVerified
	ClockdriftLevelNumCategories
)

// ClockdriftDetector detects clockdrift by analyzing the estimated delay
// (clockdrift_detector.h).
type ClockdriftDetector struct {
	delayHistory     [3]int
	level            ClockdriftDetectorLevel
	stabilityCounter int
}

// NewClockdriftDetector returns a detector at level None.
func NewClockdriftDetector() *ClockdriftDetector { return &ClockdriftDetector{} }

// ClockdriftLevel returns the detected clockdrift level.
func (d *ClockdriftDetector) ClockdriftLevel() ClockdriftDetectorLevel { return d.level }

// Update feeds one delay estimate.
func (d *ClockdriftDetector) Update(delayEstimate int) {
	if delayEstimate == d.delayHistory[0] {
		// Reset clockdrift level if delay estimate is stable for 7500 blocks (30
		// seconds).
		d.stabilityCounter++
		if d.stabilityCounter > 7500 {
			d.level = ClockdriftLevelNone
		}
		return
	}

	d.stabilityCounter = 0
	d1 := d.delayHistory[0] - delayEstimate
	d2 := d.delayHistory[1] - delayEstimate
	d3 := d.delayHistory[2] - delayEstimate

	// Patterns recognized as positive clockdrift:
	// [x-3], x-2, x-1, x.
	// [x-3], x-1, x-2, x.
	probableDriftUp := (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1)
	driftUp := probableDriftUp && d3 == -3

	// Patterns recognized as negative clockdrift:
	// [x+3], x+2, x+1, x.
	// [x+3], x+1, x+2, x.
	probableDriftDown := (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1)
	driftDown := probableDriftDown && d3 == 3

	// Set clockdrift level.
	if driftUp || driftDown {
		d.level = ClockdriftLevelVerified
	} else if (probableDriftUp || probableDriftDown) && d.level == ClockdriftLevelNone {
		d.level = ClockdriftLevelProbable
	}

	// Shift delay history one step.
	d.delayHistory[2] = d.delayHistory[1]
	d.delayHistory[1] = d.delayHistory[0]
	d.delayHistory[0] = delayEstimate
}
