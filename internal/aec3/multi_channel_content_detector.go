package aec3

const supNumFramesPerSecond = 100

func supHasStereoContent(frame Frame, detectionThreshold float32) bool {
	if frame.NumChannels() < 2 {
		return false
	}
	for band := 0; band < frame.NumBands(); band++ {
		l, r := frame.View(band, 0), frame.View(band, 1)
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

// MultiChannelContentDetector decides whether the render signal carries
// proper multichannel content (multi_channel_content_detector.h).
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

// NewMultiChannelContentDetector mirrors the C++ constructor. With
// stereoDetectionTimeoutThresholdSeconds <= 0 a detection never times out.
func NewMultiChannelContentDetector(detectStereoContent bool, numRenderInputChannels int,
	detectionThreshold float32, stereoDetectionTimeoutThresholdSeconds int,
	stereoDetectionHysteresisSeconds float32) *MultiChannelContentDetector {
	d := &MultiChannelContentDetector{}
	d.init(detectStereoContent, numRenderInputChannels, detectionThreshold,
		stereoDetectionTimeoutThresholdSeconds, stereoDetectionHysteresisSeconds)
	return d
}

func (d *MultiChannelContentDetector) init(detectStereoContent bool, numRenderInputChannels int,
	detectionThreshold float32, stereoDetectionTimeoutThresholdSeconds int,
	stereoDetectionHysteresisSeconds float32) {
	*d = MultiChannelContentDetector{
		detectStereoContent:                   detectStereoContent,
		detectionThreshold:                    detectionThreshold,
		stereoDetectionHysteresisFrames:       int(stereoDetectionHysteresisSeconds * supNumFramesPerSecond),
		persistentMultichannelContentDetected: !detectStereoContent && numRenderInputChannels > 1,
	}
	if stereoDetectionTimeoutThresholdSeconds > 0 {
		d.hasDetectionTimeout = true
		d.detectionTimeoutThresholdFrames = stereoDetectionTimeoutThresholdSeconds * supNumFramesPerSecond
	}
}

// UpdateDetection compares the first two channels of the render frame. It
// reports whether the persistent multichannel state changed.
func (d *MultiChannelContentDetector) UpdateDetection(frame Frame) bool {
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

	if d.consecutiveFramesWithStereo > int64(d.stereoDetectionHysteresisFrames) {
		d.persistentMultichannelContentDetected = true
	}
	if d.hasDetectionTimeout &&
		d.framesSinceStereoDetectedLast >= int64(d.detectionTimeoutThresholdFrames) {
		d.persistentMultichannelContentDetected = false
	}

	d.temporaryMultichannelContentDetected = !d.persistentMultichannelContentDetected && stereoDetectedInFrame

	return previous != d.persistentMultichannelContentDetected
}

func (d *MultiChannelContentDetector) IsProperMultiChannelContentDetected() bool {
	return d.persistentMultichannelContentDetected
}

func (d *MultiChannelContentDetector) IsTemporaryMultiChannelContentDetected() bool {
	return d.temporaryMultichannelContentDetected
}
