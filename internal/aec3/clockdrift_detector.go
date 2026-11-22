package aec3

// ClockdriftDetectorLevel mirrors ClockdriftDetector::Level.
type ClockdriftDetectorLevel int

const (
	ClockdriftLevelNone ClockdriftDetectorLevel = iota
	ClockdriftLevelProbable
	ClockdriftLevelVerified
	ClockdriftLevelNumCategories
)

// ClockdriftDetector detects clock drift from the estimated delays
// (clockdrift_detector.h).
type ClockdriftDetector struct {
	delayHistory     [3]int
	level            ClockdriftDetectorLevel
	stabilityCounter int
}

func NewClockdriftDetector() *ClockdriftDetector { return &ClockdriftDetector{} }

func (d *ClockdriftDetector) ClockdriftLevel() ClockdriftDetectorLevel { return d.level }

// Update takes the next delay estimate.
func (d *ClockdriftDetector) Update(delayEstimate int) {
	if delayEstimate == d.delayHistory[0] {
		// A delay that is stable for 7500 blocks (30 seconds) resets the
		// level.
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

	// Positive drift: [x-3], x-2, x-1, x or [x-3], x-1, x-2, x.
	probableDriftUp := (d1 == -1 && d2 == -2) || (d1 == -2 && d2 == -1)
	driftUp := probableDriftUp && d3 == -3

	// Negative drift: [x+3], x+2, x+1, x or [x+3], x+1, x+2, x.
	probableDriftDown := (d1 == 1 && d2 == 2) || (d1 == 2 && d2 == 1)
	driftDown := probableDriftDown && d3 == 3

	if driftUp || driftDown {
		d.level = ClockdriftLevelVerified
	} else if (probableDriftUp || probableDriftDown) && d.level == ClockdriftLevelNone {
		d.level = ClockdriftLevelProbable
	}

	d.delayHistory[2] = d.delayHistory[1]
	d.delayHistory[1] = d.delayHistory[0]
	d.delayHistory[0] = delayEstimate
}
