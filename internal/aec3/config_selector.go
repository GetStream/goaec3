package aec3

// ConfigSelector selects the config to use (config_selector.h). It holds no
// pointers.
type ConfigSelector struct {
	config                EchoCanceller3Config
	multichannelConfig    EchoCanceller3Config
	hasMultichannelConfig bool
	multichannelActive    bool
}

// NewConfigSelector mirrors the C++ constructor. A nil multichannelConfig is
// std::nullopt.
func NewConfigSelector(config, multichannelConfig *EchoCanceller3Config,
	numRenderInputChannels int) *ConfigSelector {
	s := &ConfigSelector{}
	s.init(config, multichannelConfig, numRenderInputChannels)
	return s
}

func (s *ConfigSelector) init(config, multichannelConfig *EchoCanceller3Config, numRenderInputChannels int) {
	*s = ConfigSelector{config: *config}
	if multichannelConfig != nil {
		s.multichannelConfig = *multichannelConfig
		s.hasMultichannelConfig = true
	}
	s.Update(!s.config.MultiChannel.DetectStereoContent && numRenderInputChannels > 1)
}

// Update selects the config for content that is multichannel or not.
func (s *ConfigSelector) Update(multichannelContent bool) {
	s.multichannelActive = multichannelContent && s.hasMultichannelConfig
}

// ActiveConfig returns the active config. The pointer and its target stay
// valid for the lifetime of the selector.
func (s *ConfigSelector) ActiveConfig() *EchoCanceller3Config {
	if s.multichannelActive {
		return &s.multichannelConfig
	}
	return &s.config
}
