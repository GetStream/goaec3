package aec3

import "math"

// EchoCanceller3Config mirrors api/audio/echo_canceller3_config.h. Field
// names are the C++ snake_case names in CamelCase; size_t fields are int.
type EchoCanceller3Config struct {
	Buffering struct {
		ExcessRenderDetectionIntervalBlocks int
		MaxAllowedExcessRenderBlocks        int
	}
	Delay struct {
		DefaultDelay                     int
		DownSamplingFactor               int
		NumFilters                       int
		DelayHeadroomSamples             int
		HysteresisLimitBlocks            int
		FixedCaptureDelaySamples         int
		DelayEstimateSmoothing           float32
		DelayEstimateSmoothingDelayFound float32
		DelayCandidateDetectionThreshold float32
		DelaySelectionThresholds         struct{ Initial, Converged int }
		UseExternalDelayEstimator        bool
		LogWarningOnDelayChanges         bool
		RenderAlignmentMixing            AlignmentMixing
		CaptureAlignmentMixing           AlignmentMixing
		DetectPreEcho                    bool
	}
	Filter struct {
		Refined                       RefinedConfiguration
		Coarse                        CoarseConfiguration
		RefinedInitial                RefinedConfiguration
		CoarseInitial                 CoarseConfiguration
		ConfigChangeDurationBlocks    int
		InitialStateSeconds           float32
		CoarseResetHangoverBlocks     int
		ConservativeInitialPhase      bool
		EnableCoarseFilterOutputUsage bool
		UseLinearFilter               bool
		HighPassFilterEchoReference   bool
		ExportLinearAecOutput         bool
	}
	Erle struct {
		Min                        float32
		MaxL                       float32
		MaxH                       float32
		OnsetDetection             bool
		NumSections                int
		ClampQualityEstimateToZero bool
		ClampQualityEstimateToOne  bool
	}
	EpStrength struct {
		DefaultGain                            float32
		DefaultLen                             float32
		NearendLen                             float32
		EchoCanSaturate                        bool
		BoundedErl                             bool
		ErleOnsetCompensationInDominantNearend bool
		UseConservativeTailFrequencyResponse   bool
	}
	EchoAudibility struct {
		LowRenderLimit                  float32
		NormalRenderLimit               float32
		FloorPower                      float32
		AudibilityThresholdLf           float32
		AudibilityThresholdMf           float32
		AudibilityThresholdHf           float32
		UseStationarityProperties       bool
		UseStationarityPropertiesAtInit bool
	}
	RenderLevels struct {
		ActiveRenderLimit            float32
		PoorExcitationRenderLimit    float32
		PoorExcitationRenderLimitDs8 float32
		RenderPowerGainDb            float32
	}
	EchoRemovalControl struct {
		HasClockDrift           bool
		LinearAndStableEchoPath bool
	}
	EchoModel struct {
		NoiseFloorHold             int
		MinNoiseFloorPower         float32
		StationaryGateSlope        float32
		NoiseGatePower             float32
		NoiseGateSlope             float32
		RenderPreWindowSize        int
		RenderPostWindowSize       int
		ModelReverbInNonlinearMode bool
	}
	ComfortNoise struct {
		NoiseFloorDbfs float32
	}
	Suppressor struct {
		NearendAverageBlocks          int
		NormalTuning                  Tuning
		NearendTuning                 Tuning
		LfSmoothingDuringInitialPhase bool
		LastPermanentLfSmoothingBand  int
		LastLfSmoothingBand           int
		LastLfBand                    int
		FirstHfBand                   int
		DominantNearendDetection      DominantNearendDetectionConfig
		SubbandNearendDetection       SubbandNearendDetectionConfig
		UseSubbandNearendDetection    bool
		HighBandsSuppression          struct {
			EnrThreshold                   float32
			MaxGainDuringEcho              float32
			AntiHowlingActivationThreshold float32
			AntiHowlingGain                float32
		}
		FloorFirstIncrease        float32
		ConservativeHfSuppression bool
	}
	MultiChannel struct {
		DetectStereoContent                    bool
		StereoDetectionThreshold               float32
		StereoDetectionTimeoutThresholdSeconds int
		StereoDetectionHysteresisSeconds       float32
	}
}

type AlignmentMixing struct {
	Downmix                bool
	AdaptiveSelection      bool
	ActivityPowerThreshold float32
	PreferFirstTwoChannels bool
}

type RefinedConfiguration struct {
	LengthBlocks     int
	LeakageConverged float32
	LeakageDiverged  float32
	ErrorFloor       float32
	ErrorCeil        float32
	NoiseGate        float32
}

type CoarseConfiguration struct {
	LengthBlocks int
	Rate         float32
	NoiseGate    float32
}

type MaskingThresholds struct {
	EnrTransparent float32
	EnrSuppress    float32
	EmrTransparent float32
}

type Tuning struct {
	MaskLf         MaskingThresholds
	MaskHf         MaskingThresholds
	MaxIncFactor   float32
	MaxDecFactorLf float32
}

type SubbandRegion struct{ Low, High int }

// DefaultEchoCanceller3Config returns the C++ default-constructed config.
func DefaultEchoCanceller3Config() EchoCanceller3Config {
	var c EchoCanceller3Config
	c.Buffering.ExcessRenderDetectionIntervalBlocks = 250
	c.Buffering.MaxAllowedExcessRenderBlocks = 8

	d := &c.Delay
	d.DefaultDelay = 5
	d.DownSamplingFactor = 4
	d.NumFilters = 5
	d.DelayHeadroomSamples = 32
	d.HysteresisLimitBlocks = 1
	d.DelayEstimateSmoothing = 0.7
	d.DelayEstimateSmoothingDelayFound = 0.7
	d.DelayCandidateDetectionThreshold = 0.2
	d.DelaySelectionThresholds.Initial = 5
	d.DelaySelectionThresholds.Converged = 20
	d.RenderAlignmentMixing = AlignmentMixing{false, true, 10000, true}
	d.CaptureAlignmentMixing = AlignmentMixing{false, true, 10000, false}
	d.DetectPreEcho = true

	f := &c.Filter
	f.Refined = RefinedConfiguration{13, 0.00005, 0.05, 0.001, 2, 20075344}
	f.Coarse = CoarseConfiguration{13, 0.7, 20075344}
	f.RefinedInitial = RefinedConfiguration{12, 0.005, 0.5, 0.001, 2, 20075344}
	f.CoarseInitial = CoarseConfiguration{12, 0.9, 20075344}
	f.ConfigChangeDurationBlocks = 250
	f.InitialStateSeconds = 2.5
	f.CoarseResetHangoverBlocks = 25
	f.EnableCoarseFilterOutputUsage = true
	f.UseLinearFilter = true

	c.Erle.Min = 1
	c.Erle.MaxL = 4
	c.Erle.MaxH = 1.5
	c.Erle.OnsetDetection = true
	c.Erle.NumSections = 1
	c.Erle.ClampQualityEstimateToZero = true
	c.Erle.ClampQualityEstimateToOne = true

	c.EpStrength.DefaultGain = 1
	c.EpStrength.DefaultLen = 0.83
	c.EpStrength.NearendLen = 0.83
	c.EpStrength.EchoCanSaturate = true
	c.EpStrength.UseConservativeTailFrequencyResponse = true

	c.EchoAudibility.LowRenderLimit = 4 * 64
	c.EchoAudibility.NormalRenderLimit = 64
	c.EchoAudibility.FloorPower = 2 * 64
	c.EchoAudibility.AudibilityThresholdLf = 10
	c.EchoAudibility.AudibilityThresholdMf = 10
	c.EchoAudibility.AudibilityThresholdHf = 10

	c.RenderLevels.ActiveRenderLimit = 100
	c.RenderLevels.PoorExcitationRenderLimit = 150
	c.RenderLevels.PoorExcitationRenderLimitDs8 = 20

	m := &c.EchoModel
	m.NoiseFloorHold = 50
	m.MinNoiseFloorPower = 1638400
	m.StationaryGateSlope = 10
	m.NoiseGatePower = 27509.42
	m.NoiseGateSlope = 0.3
	m.RenderPreWindowSize = 1
	m.RenderPostWindowSize = 1
	m.ModelReverbInNonlinearMode = true

	c.ComfortNoise.NoiseFloorDbfs = -96.03406

	s := &c.Suppressor
	s.NearendAverageBlocks = 4
	s.NormalTuning = Tuning{MaskingThresholds{.3, .4, .3}, MaskingThresholds{.07, .1, .3}, 2.0, 0.25}
	s.NearendTuning = Tuning{MaskingThresholds{1.09, 1.1, .3}, MaskingThresholds{.1, .3, .3}, 2.0, 0.25}
	s.LfSmoothingDuringInitialPhase = true
	s.LastPermanentLfSmoothingBand = 0
	s.LastLfSmoothingBand = 5
	s.LastLfBand = 5
	s.FirstHfBand = 8
	dn := &s.DominantNearendDetection
	dn.EnrThreshold = .25
	dn.EnrExitThreshold = 10
	dn.SnrThreshold = 30
	dn.HoldDuration = 50
	dn.TriggerThreshold = 12
	dn.UseDuringInitialPhase = true
	dn.UseUnboundedEchoSpectrum = true
	sn := &s.SubbandNearendDetection
	sn.NearendAverageBlocks = 1
	sn.Subband1 = SubbandRegion{1, 1}
	sn.Subband2 = SubbandRegion{1, 1}
	sn.NearendThreshold = 1
	sn.SnrThreshold = 1
	s.HighBandsSuppression.EnrThreshold = 1
	s.HighBandsSuppression.MaxGainDuringEcho = 1
	s.HighBandsSuppression.AntiHowlingActivationThreshold = 400
	s.HighBandsSuppression.AntiHowlingGain = 1
	s.FloorFirstIncrease = 0.00001

	c.MultiChannel.DetectStereoContent = true
	c.MultiChannel.StereoDetectionThreshold = 0
	c.MultiChannel.StereoDetectionTimeoutThresholdSeconds = 300
	c.MultiChannel.StereoDetectionHysteresisSeconds = 2
	return c
}

// CreateDefaultMultichannelConfig mirrors the C++ static of the same name.
func CreateDefaultMultichannelConfig() EchoCanceller3Config {
	c := DefaultEchoCanceller3Config()
	c.Filter.Coarse.LengthBlocks = 11
	c.Filter.Coarse.Rate = 0.95
	c.Filter.CoarseInitial.LengthBlocks = 11
	c.Filter.CoarseInitial.Rate = 0.95
	c.Suppressor.NormalTuning.MaxDecFactorLf = 0.35
	c.Suppressor.NormalTuning.MaxIncFactor = 1.5
	return c
}

func limitF(v *float32, lo, hi float32) bool {
	c := *v
	if c < lo {
		c = lo
	} else if c > hi {
		c = hi
	}
	if math.IsNaN(float64(c)) || math.IsInf(float64(c), 0) {
		c = lo
	}
	res := *v == c
	*v = c
	return res
}

func limitI(v *int, lo, hi int) bool {
	c := min(max(*v, lo), hi)
	res := *v == c
	*v = c
	return res
}

func floorLimit(v *int, lo int) bool {
	c := max(*v, lo)
	res := *v == c
	*v = c
	return res
}

// Validate clamps c into its legal ranges and reports whether it was valid.
func (c *EchoCanceller3Config) Validate() bool {
	res := true
	and := func(ok bool) { res = res && ok }
	if c.Delay.DownSamplingFactor != 4 && c.Delay.DownSamplingFactor != 8 {
		c.Delay.DownSamplingFactor = 4
		res = false
	}
	and(limitI(&c.Delay.DefaultDelay, 0, 5000))
	and(limitI(&c.Delay.NumFilters, 0, 5000))
	and(limitI(&c.Delay.DelayHeadroomSamples, 0, 5000))
	and(limitI(&c.Delay.HysteresisLimitBlocks, 0, 5000))
	and(limitI(&c.Delay.FixedCaptureDelaySamples, 0, 5000))
	and(limitF(&c.Delay.DelayEstimateSmoothing, 0, 1))
	and(limitF(&c.Delay.DelayCandidateDetectionThreshold, 0, 1))
	and(limitI(&c.Delay.DelaySelectionThresholds.Initial, 1, 250))
	and(limitI(&c.Delay.DelaySelectionThresholds.Converged, 1, 250))

	for _, r := range []*RefinedConfiguration{&c.Filter.Refined, &c.Filter.RefinedInitial} {
		and(floorLimit(&r.LengthBlocks, 1))
		and(limitF(&r.LeakageConverged, 0, 1000))
		and(limitF(&r.LeakageDiverged, 0, 1000))
		and(limitF(&r.ErrorFloor, 0, 1000))
		and(limitF(&r.ErrorCeil, 0, 100000000))
		and(limitF(&r.NoiseGate, 0, 100000000))
	}
	if c.Filter.Refined.LengthBlocks < c.Filter.RefinedInitial.LengthBlocks {
		c.Filter.RefinedInitial.LengthBlocks = c.Filter.Refined.LengthBlocks
		res = false
	}
	for _, r := range []*CoarseConfiguration{&c.Filter.Coarse, &c.Filter.CoarseInitial} {
		and(floorLimit(&r.LengthBlocks, 1))
		and(limitF(&r.Rate, 0, 1))
		and(limitF(&r.NoiseGate, 0, 100000000))
	}
	if c.Filter.Coarse.LengthBlocks < c.Filter.CoarseInitial.LengthBlocks {
		c.Filter.CoarseInitial.LengthBlocks = c.Filter.Coarse.LengthBlocks
		res = false
	}
	and(limitI(&c.Filter.ConfigChangeDurationBlocks, 0, 100000))
	and(limitF(&c.Filter.InitialStateSeconds, 0, 100))
	and(limitI(&c.Filter.CoarseResetHangoverBlocks, 0, 250000))

	and(limitF(&c.Erle.Min, 1, 100000))
	and(limitF(&c.Erle.MaxL, 1, 100000))
	and(limitF(&c.Erle.MaxH, 1, 100000))
	if c.Erle.Min > c.Erle.MaxL || c.Erle.Min > c.Erle.MaxH {
		c.Erle.Min = min(c.Erle.MaxL, c.Erle.MaxH)
		res = false
	}
	and(limitI(&c.Erle.NumSections, 1, c.Filter.Refined.LengthBlocks))

	and(limitF(&c.EpStrength.DefaultGain, 0, 1000000))
	and(limitF(&c.EpStrength.DefaultLen, -1, 1))
	and(limitF(&c.EpStrength.NearendLen, -1, 1))

	const maxPow = 32768 * 32768
	ea := &c.EchoAudibility
	and(limitF(&ea.LowRenderLimit, 0, maxPow))
	and(limitF(&ea.NormalRenderLimit, 0, maxPow))
	and(limitF(&ea.FloorPower, 0, maxPow))
	and(limitF(&ea.AudibilityThresholdLf, 0, maxPow))
	and(limitF(&ea.AudibilityThresholdMf, 0, maxPow))
	and(limitF(&ea.AudibilityThresholdHf, 0, maxPow))

	and(limitF(&c.RenderLevels.ActiveRenderLimit, 0, maxPow))
	and(limitF(&c.RenderLevels.PoorExcitationRenderLimit, 0, maxPow))
	and(limitF(&c.RenderLevels.PoorExcitationRenderLimitDs8, 0, maxPow))

	em := &c.EchoModel
	and(limitI(&em.NoiseFloorHold, 0, 1000))
	and(limitF(&em.MinNoiseFloorPower, 0, 2000000))
	and(limitF(&em.StationaryGateSlope, 0, 1000000))
	and(limitF(&em.NoiseGatePower, 0, 1000000))
	and(limitF(&em.NoiseGateSlope, 0, 1000000))
	and(limitI(&em.RenderPreWindowSize, 0, 100))
	and(limitI(&em.RenderPostWindowSize, 0, 100))

	and(limitF(&c.ComfortNoise.NoiseFloorDbfs, -200, 0))

	s := &c.Suppressor
	and(limitI(&s.NearendAverageBlocks, 1, 5000))
	for _, t := range []*Tuning{&s.NormalTuning, &s.NearendTuning} {
		and(limitF(&t.MaskLf.EnrTransparent, 0, 100))
		and(limitF(&t.MaskLf.EnrSuppress, 0, 100))
		and(limitF(&t.MaskLf.EmrTransparent, 0, 100))
		and(limitF(&t.MaskHf.EnrTransparent, 0, 100))
		and(limitF(&t.MaskHf.EnrSuppress, 0, 100))
		and(limitF(&t.MaskHf.EmrTransparent, 0, 100))
		and(limitF(&t.MaxIncFactor, 0, 100))
		and(limitF(&t.MaxDecFactorLf, 0, 100))
	}
	and(limitI(&s.LastPermanentLfSmoothingBand, 0, 64))
	and(limitI(&s.LastLfSmoothingBand, 0, 64))
	and(limitI(&s.LastLfBand, 0, 63))
	and(limitI(&s.FirstHfBand, s.LastLfBand+1, 64))

	dn := &s.DominantNearendDetection
	and(limitF(&dn.EnrThreshold, 0, 1000000))
	and(limitF(&dn.SnrThreshold, 0, 1000000))
	and(limitI(&dn.HoldDuration, 0, 10000))
	and(limitI(&dn.TriggerThreshold, 0, 10000))

	sn := &s.SubbandNearendDetection
	and(limitI(&sn.NearendAverageBlocks, 1, 1024))
	and(limitI(&sn.Subband1.Low, 0, 65))
	and(limitI(&sn.Subband1.High, sn.Subband1.Low, 65))
	and(limitI(&sn.Subband2.Low, 0, 65))
	and(limitI(&sn.Subband2.High, sn.Subband2.Low, 65))
	and(limitF(&sn.NearendThreshold, 0, 1e24))
	and(limitF(&sn.SnrThreshold, 0, 1e24))

	hb := &s.HighBandsSuppression
	and(limitF(&hb.EnrThreshold, 0, 1000000))
	and(limitF(&hb.MaxGainDuringEcho, 0, 1))
	and(limitF(&hb.AntiHowlingActivationThreshold, 0, maxPow))
	and(limitF(&hb.AntiHowlingGain, 0, 1))

	and(limitF(&s.FloorFirstIncrease, 0, 1000000))
	return res
}
