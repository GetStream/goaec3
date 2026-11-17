package aec3

const supNumFramesPerSecond = 100

func supHasStereoContent(frame [][][]float32, detectionThreshold float32) bool {
	if len(frame[0]) < 2 {
		return false
	}
	for band := range frame {
		l, r := frame[band][0], frame[band][1]
		for k := range l {
			d := l[k] - r[k]
			if d < 0 {
				d = -d
			}
			if d > detectionThreshold {
				return true
			}
		}
	}
	return false
}

// MultiChannelContentDetector analyzes the render signal to decide whether it
// carries proper multichannel content.
type MultiChannelContentDetector struct {
	detectStereoContent                   bool
	detectionThreshold                    float32
	hasDetectionTimeout                   bool
	detectionTimeoutThresholdFrames       int
	stereoDetectionHysteresisFrames       int
	persistentMultichannelContentDetected bool
	temporaryMultichannelContentDetected  bool
	framesSinceStereoDetectedLast         int64
	consecutiveFramesWithStereo           int64
}

// NewMultiChannelContentDetector creates a detector. If
// stereoDetectionTimeoutThresholdSeconds <= 0, no timeout is applied: once
// multichannel is detected, the detector remains in that state for its
// lifetime.
func NewMultiChannelContentDetector(detectStereoContent bool, numRenderInputChannels int,
	detectionThreshold float32, stereoDetectionTimeoutThresholdSeconds int,
	stereoDetectionHysteresisSeconds float32) *MultiChannelContentDetector {
	d := &MultiChannelContentDetector{
		detectStereoContent:                   detectStereoContent,
		detectionThreshold:                    detectionThreshold,
		stereoDetectionHysteresisFrames:       int(stereoDetectionHysteresisSeconds * supNumFramesPerSecond),
		persistentMultichannelContentDetected: !detectStereoContent && numRenderInputChannels > 1,
	}
	if stereoDetectionTimeoutThresholdSeconds > 0 {
		d.hasDetectionTimeout = true
		d.detectionTimeoutThresholdFrames = stereoDetectionTimeoutThresholdSeconds * supNumFramesPerSecond
	}
	return d
}

// UpdateDetection compares the left and right channels in the render frame
// ([band][channel][sample]) to determine whether the signal is a proper
// multichannel signal. It returns whether the persistent multichannel state
// changed.
func (d *MultiChannelContentDetector) UpdateDetection(frame [][][]float32) bool {
	if !d.detectStereoContent {
		return false
	}

	previous := d.persistentMultichannelContentDetected
	stereoDetectedInFrame := supHasStereoContent(frame, d.detectionThreshold)

	if stereoDetectedInFrame {
		d.consecutiveFramesWithStereo++
		d.framesSinceStereoDetectedLast = 0
	} else {
		d.consecutiveFramesWithStereo = 0
		d.framesSinceStereoDetectedLast++
	}

	// Detect persistent multichannel content.
	if d.consecutiveFramesWithStereo > int64(d.stereoDetectionHysteresisFrames) {
		d.persistentMultichannelContentDetected = true
	}
	if d.hasDetectionTimeout &&
		d.framesSinceStereoDetectedLast >= int64(d.detectionTimeoutThresholdFrames) {
		d.persistentMultichannelContentDetected = false
	}

	// Detect temporary multichannel content.
	d.temporaryMultichannelContentDetected = !d.persistentMultichannelContentDetected && stereoDetectedInFrame

	return previous != d.persistentMultichannelContentDetected
}

func (d *MultiChannelContentDetector) IsProperMultiChannelContentDetected() bool {
	return d.persistentMultichannelContentDetected
}

func (d *MultiChannelContentDetector) IsTemporaryMultiChannelContentDetected() bool {
	return d.temporaryMultichannelContentDetected
}
