package aec3

// ConfigSelector selects the config to use (config_selector.h). It holds no
// pointers, so it can be embedded and copied by value.
type ConfigSelector struct {
	config                EchoCanceller3Config
	multichannelConfig    EchoCanceller3Config
	hasMultichannelConfig bool
	multichannelActive    bool
}

// NewConfigSelector mirrors the C++ constructor; multichannelConfig is nil
// for std::nullopt.
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

// Update selects the active config based on whether multichannel content is
// detected.
func (s *ConfigSelector) Update(multichannelContent bool) {
	s.multichannelActive = multichannelContent && s.hasMultichannelConfig
}

// ActiveConfig returns the active config. The pointer stays valid, and its
// target unchanged, for the lifetime of the selector.
func (s *ConfigSelector) ActiveConfig() *EchoCanceller3Config {
	if s.multichannelActive {
		return &s.multichannelConfig
	}
	return &s.config
}
