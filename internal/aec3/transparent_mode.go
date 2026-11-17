package aec3

const (
	stBlocksSinceConvergencedFilterInit          = 10000
	stBlocksSinceConsistentEstimateInit          = 10000
	stInitialTransparentStateProbability float32 = 0.2
)

// TransparentMode mirrors transparent_mode.h: detects and toggles the
// transparent mode, in which the suppressor applies less suppression.
type TransparentMode interface {
	// Active reports whether the transparent mode should be active.
	Active() bool
	// Reset resets the state of the detector.
	Reset()
	// Update updates the detection decision based on new data.
	Update(filterDelayBlocks int, anyFilterConsistent, anyFilterConverged, anyCoarseFilterConverged,
		allFiltersDiverged, activeRender, saturatedCapture bool)
}

// CreateTransparentMode mirrors TransparentMode::Create with field trials at
// their defaults: nil when ep_strength.bounded_erl is set, otherwise the
// legacy classifier.
func CreateTransparentMode(config *EchoCanceller3Config) TransparentMode {
	if config.EpStrength.BoundedErl {
		return nil
	}
	return NewLegacyTransparentMode(config)
}

// TransparentModeHmm is the HMM classifier (TransparentModeImpl), selected
// in C++ by the WebRTC-Aec3TransparentModeHmm field trial.
type TransparentModeHmm struct {
	transparencyActivated bool
	probTransparentState  float32
}

func NewTransparentModeHmm() *TransparentModeHmm {
	return &TransparentModeHmm{probTransparentState: stInitialTransparentStateProbability}
}

func (t *TransparentModeHmm) Active() bool { return t.transparencyActivated }

func (t *TransparentModeHmm) Reset() {
	t.transparencyActivated = false
	t.probTransparentState = stInitialTransparentStateProbability
}

var (
	stHmmA = [2]float32{0.000001, stSub(1, 0.000001)}
	stHmmB = [2][2]float32{
		{stSub(1, 0.01), 0.01},
		{stSub(1, 0.001), 0.001},
	}
)

func (t *TransparentModeHmm) Update(_ int, _, _, anyCoarseFilterConverged, _, activeRender, _ bool) {
	if !activeRender {
		return
	}
	probTransparent := t.probTransparentState
	probNormal := 1 - probTransparent
	probTransitionTransparent := float32(probNormal*stHmmA[0]) + float32(probTransparent*stHmmA[1])
	probTransitionNormal := 1 - probTransitionTransparent
	out := 0
	if anyCoarseFilterConverged {
		out = 1
	}
	probJointNormal := probTransitionNormal * stHmmB[0][out]
	probJointTransparent := probTransitionTransparent * stHmmB[1][out]
	t.probTransparentState = probJointTransparent / (probJointNormal + probJointTransparent)
	if t.probTransparentState > 0.95 {
		t.transparencyActivated = true
	} else if t.probTransparentState < 0.5 {
		t.transparencyActivated = false
	}
}

// LegacyTransparentMode is the legacy classifier (LegacyTransparentModeImpl).
type LegacyTransparentMode struct {
	linearAndStableEchoPath         bool
	captureBlockCounter             int
	transparencyActivated           bool
	activeBlocksSinceSaneFilter     int
	saneFilterObserved              bool
	finiteErlRecentlyDetected       bool
	nonConvergedSequenceSize        int
	divergedSequenceSize            int
	activeNonConvergedSequenceSize  int
	numConvergedBlocks              int
	recentConvergenceDuringActivity bool
	strongNotSaturatedRenderBlocks  int
}

func NewLegacyTransparentMode(config *EchoCanceller3Config) *LegacyTransparentMode {
	return &LegacyTransparentMode{
		linearAndStableEchoPath:     config.EchoRemovalControl.LinearAndStableEchoPath,
		activeBlocksSinceSaneFilter: stBlocksSinceConsistentEstimateInit,
		nonConvergedSequenceSize:    stBlocksSinceConvergencedFilterInit,
	}
}

func (t *LegacyTransparentMode) Active() bool { return t.transparencyActivated }

func (t *LegacyTransparentMode) Reset() {
	t.nonConvergedSequenceSize = stBlocksSinceConvergencedFilterInit
	t.divergedSequenceSize = 0
	t.strongNotSaturatedRenderBlocks = 0
	if t.linearAndStableEchoPath {
		t.recentConvergenceDuringActivity = false
	}
}

func (t *LegacyTransparentMode) Update(filterDelayBlocks int, anyFilterConsistent, anyFilterConverged, _,
	allFiltersDiverged, activeRender, saturatedCapture bool) {
	t.captureBlockCounter++
	if activeRender && !saturatedCapture {
		t.strongNotSaturatedRenderBlocks++
	}
	if anyFilterConsistent && filterDelayBlocks < 5 {
		t.saneFilterObserved = true
		t.activeBlocksSinceSaneFilter = 0
	} else if activeRender {
		t.activeBlocksSinceSaneFilter++
	}
	var saneFilterRecentlySeen bool
	if !t.saneFilterObserved {
		saneFilterRecentlySeen = t.captureBlockCounter <= 5*NumBlocksPerSecond
	} else {
		saneFilterRecentlySeen = t.activeBlocksSinceSaneFilter <= 30*NumBlocksPerSecond
	}
	if anyFilterConverged {
		t.recentConvergenceDuringActivity = true
		t.activeNonConvergedSequenceSize = 0
		t.nonConvergedSequenceSize = 0
		t.numConvergedBlocks++
	} else {
		t.nonConvergedSequenceSize++
		if t.nonConvergedSequenceSize > 20*NumBlocksPerSecond {
			t.numConvergedBlocks = 0
		}
		if activeRender {
			t.activeNonConvergedSequenceSize++
			if t.activeNonConvergedSequenceSize > 60*NumBlocksPerSecond {
				t.recentConvergenceDuringActivity = false
			}
		}
	}
	if !allFiltersDiverged {
		t.divergedSequenceSize = 0
	} else {
		t.divergedSequenceSize++
		if t.divergedSequenceSize >= 60 {
			t.nonConvergedSequenceSize = stBlocksSinceConvergencedFilterInit
		}
	}
	if t.activeNonConvergedSequenceSize > 60*NumBlocksPerSecond {
		t.finiteErlRecentlyDetected = false
	}
	if t.numConvergedBlocks > 50 {
		t.finiteErlRecentlyDetected = true
	}
	if t.finiteErlRecentlyDetected {
		t.transparencyActivated = false
	} else if saneFilterRecentlySeen && t.recentConvergenceDuringActivity {
		t.transparencyActivated = false
	} else {
		filterShouldHaveConverged := t.strongNotSaturatedRenderBlocks > 6*NumBlocksPerSecond
		t.transparencyActivated = filterShouldHaveConverged
	}
}
