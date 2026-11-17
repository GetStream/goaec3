package aec3

// ConfigSelector selects between the mono and the multichannel config.
type ConfigSelector struct {
	config                EchoCanceller3Config
	multichannelConfig    EchoCanceller3Config
	hasMultichannelConfig bool
	activeConfig          *EchoCanceller3Config
}

// NewConfigSelector copies config and, when non-nil, multichannelConfig
// (nil corresponds to std::nullopt). The two configs must agree on the
// delay, linear-output, high-pass and stereo-detection settings (a debug-only
// check in the C++).
func NewConfigSelector(config, multichannelConfig *EchoCanceller3Config,
	numRenderInputChannels int) *ConfigSelector {
	s := &ConfigSelector{config: *config}
	if multichannelConfig != nil {
		s.multichannelConfig = *multichannelConfig
		s.hasMultichannelConfig = true
	}
	s.Update(!s.config.MultiChannel.DetectStereoContent && numRenderInputChannels > 1)
	return s
}

// Update updates the config selection based on the detection of multichannel
// content.
func (s *ConfigSelector) Update(multichannelContent bool) {
	if multichannelContent && s.hasMultichannelConfig {
		s.activeConfig = &s.multichannelConfig
	} else {
		s.activeConfig = &s.config
	}
}

// ActiveConfig returns the selected config.
func (s *ConfigSelector) ActiveConfig() *EchoCanceller3Config { return s.activeConfig }
