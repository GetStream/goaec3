// Package goaec3 is a pure-Go acoustic echo canceller: a port of WebRTC's
// AEC3 (libwebrtc m141) that reproduces its output bit-exactly.
//
// An EchoCanceller consumes 10 ms frames of full-band audio. Feed it the
// signal played to the loudspeaker (the render, or far-end, signal) with
// AnalyzeRender and the microphone signal (the capture, or near-end, signal)
// with ProcessCapture, which removes the echo of the render signal from the
// capture frame in place.
//
// Samples are float32 in [-1, 1], laid out as frame[channel][sample] with
// sampleRateHz/100 samples per channel, like the float interface of the
// WebRTC AudioProcessing module. Internally the signal is converted to the
// 16-bit float scale AEC3 works in and, at 32 and 48 kHz, split into 16 kHz
// bands and merged back, exactly as webrtc::AudioBuffer does.
//
// Once warmed up, AnalyzeRender and ProcessCapture do not allocate.
package goaec3

import (
	"errors"

	"github.com/GetStream/goaec3/internal/aec3"
	"github.com/GetStream/goaec3/internal/bands"
)

// Config is the AEC3 configuration (webrtc::EchoCanceller3Config). Fields
// are the C++ snake_case names in CamelCase.
type Config = aec3.EchoCanceller3Config

// DefaultConfig returns the default configuration, equal to a
// default-constructed webrtc::EchoCanceller3Config.
func DefaultConfig() Config { return aec3.DefaultEchoCanceller3Config() }

// MultichannelConfig returns the default configuration for multichannel
// render content (EchoCanceller3Config::CreateDefaultMultichannelConfig).
func MultichannelConfig() Config { return aec3.CreateDefaultMultichannelConfig() }

// Metrics are the echo canceller metrics (webrtc::EchoControl::Metrics).
type Metrics struct {
	// EchoReturnLoss is the echo return loss in dB.
	EchoReturnLoss float64
	// EchoReturnLossEnhancement is the echo return loss enhancement in dB.
	EchoReturnLossEnhancement float64
	// DelayMs is the estimated render-to-capture delay in milliseconds.
	DelayMs int
}

// LinearOutputFrameSize is the number of samples per channel of the linear
// AEC output: one 10 ms frame at 16 kHz, whatever the capture rate.
const LinearOutputFrameSize = aec3.SplitBandSize

var (
	// ErrSampleRate is returned by New for a sample rate other than 16000,
	// 32000 or 48000 Hz.
	ErrSampleRate = errors.New("goaec3: sample rate must be 16000, 32000 or 48000 Hz")
	// ErrNumChannels is returned by New for a channel count below one.
	ErrNumChannels = errors.New("goaec3: channel counts must be at least 1")
	// ErrChannelCount is returned when a frame has the wrong number of
	// channels.
	ErrChannelCount = errors.New("goaec3: frame has the wrong number of channels")
	// ErrFrameLength is returned when a channel does not hold exactly one
	// 10 ms frame.
	ErrFrameLength = errors.New("goaec3: channel does not hold one 10 ms frame")
	// ErrLinearOutputDisabled is returned by ProcessCaptureWithLinearOutput
	// when Filter.ExportLinearAecOutput is not set in the config.
	ErrLinearOutputDisabled = errors.New("goaec3: linear output requires Filter.ExportLinearAecOutput")
)

// ec3SideBuffers holds the FloatS16 full-band and split-band copies of one
// signal direction.
type ec3SideBuffers struct {
	splitter *bands.BandSplitter
	full     [][]float32   // [channel][frameLength]
	split    [][][]float32 // [band][channel][SplitBandSize]
}

func ec3NewSideBuffers(sampleRateHz, numChannels int) ec3SideBuffers {
	s := ec3SideBuffers{splitter: bands.NewBandSplitter(sampleRateHz, numChannels)}
	frameLength := sampleRateHz / 100
	numBands := s.splitter.NumBands()
	s.full = make([][]float32, numChannels)
	for ch := range s.full {
		s.full[ch] = make([]float32, frameLength)
	}
	s.split = make([][][]float32, numBands)
	for band := range s.split {
		s.split[band] = make([][]float32, numChannels)
		for ch := range s.split[band] {
			if numBands == 1 {
				// At 16 kHz the single band is the full-band signal.
				s.split[band][ch] = s.full[ch]
			} else {
				s.split[band][ch] = make([]float32, aec3.SplitBandSize)
			}
		}
	}
	return s
}

func (s *ec3SideBuffers) splitIntoFrequencyBands() {
	if s.splitter.NumBands() > 1 {
		s.splitter.SplitIntoFrequencyBands(s.full, s.split)
	}
}

func (s *ec3SideBuffers) mergeFrequencyBands() {
	if s.splitter.NumBands() > 1 {
		s.splitter.MergeFrequencyBands(s.split, s.full)
	}
}

func ec3CheckFrame(frame [][]float32, numChannels, frameLength int) error {
	if len(frame) != numChannels {
		return ErrChannelCount
	}
	for _, ch := range frame {
		if len(ch) != frameLength {
			return ErrFrameLength
		}
	}
	return nil
}

// EchoCanceller is an AEC3 echo canceller for one render stream and one
// capture stream.
//
// AnalyzeRender may be called concurrently with the capture-side methods
// (ProcessCapture, ProcessCaptureWithLinearOutput, SetAudioBufferDelay,
// SetCaptureOutputUsage, Metrics). Calls on each side must be serialized.
type EchoCanceller struct {
	ec                 *aec3.EchoCanceller3
	sampleRateHz       int
	frameLength        int
	numRenderChannels  int
	numCaptureChannels int
	exportLinearOutput bool
	echoPathGainChange bool

	render  ec3SideBuffers
	capture ec3SideBuffers
	linear  [][]float32 // [channel][LinearOutputFrameSize]
}

// New creates an echo canceller for sampleRateHz (16000, 32000 or 48000)
// with the given number of render and capture channels. The config is used
// as given; call Config.Validate first to clamp it into its legal ranges.
func New(cfg Config, sampleRateHz, numRenderChannels, numCaptureChannels int) (*EchoCanceller, error) {
	return newEchoCanceller(&cfg, nil, sampleRateHz, numRenderChannels, numCaptureChannels)
}

// NewWithMultichannelConfig is like New, and switches to multichannelCfg
// while proper multichannel content is detected in the render signal (or
// from the start when cfg.MultiChannel.DetectStereoContent is false and
// there is more than one render channel). The two configs must agree on the
// delay, linear output, high-pass and stereo detection settings.
func NewWithMultichannelConfig(cfg, multichannelCfg Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int) (*EchoCanceller, error) {
	return newEchoCanceller(&cfg, &multichannelCfg, sampleRateHz, numRenderChannels, numCaptureChannels)
}

func newEchoCanceller(cfg, multichannelCfg *Config, sampleRateHz, numRenderChannels,
	numCaptureChannels int) (*EchoCanceller, error) {
	if sampleRateHz != 16000 && sampleRateHz != 32000 && sampleRateHz != 48000 {
		return nil, ErrSampleRate
	}
	if numRenderChannels < 1 || numCaptureChannels < 1 {
		return nil, ErrNumChannels
	}
	e := &EchoCanceller{
		ec:                 aec3.NewEchoCanceller3(cfg, multichannelCfg, sampleRateHz, numRenderChannels, numCaptureChannels),
		sampleRateHz:       sampleRateHz,
		frameLength:        sampleRateHz / 100,
		numRenderChannels:  numRenderChannels,
		numCaptureChannels: numCaptureChannels,
		exportLinearOutput: cfg.Filter.ExportLinearAecOutput,
		render:             ec3NewSideBuffers(sampleRateHz, numRenderChannels),
		capture:            ec3NewSideBuffers(sampleRateHz, numCaptureChannels),
	}
	if e.exportLinearOutput {
		e.linear = make([][]float32, numCaptureChannels)
		for ch := range e.linear {
			e.linear[ch] = make([]float32, LinearOutputFrameSize)
		}
	}
	return e, nil
}

// SampleRateHz returns the sample rate.
func (e *EchoCanceller) SampleRateHz() int { return e.sampleRateHz }

// FrameSize returns the number of samples per channel in a 10 ms frame.
func (e *EchoCanceller) FrameSize() int { return e.frameLength }

// AnalyzeRender buffers one 10 ms frame of the render signal,
// render[channel][sample] with NumRenderChannels channels of FrameSize
// samples in [-1, 1]. The frame is copied; the caller keeps ownership.
func (e *EchoCanceller) AnalyzeRender(render [][]float32) error {
	if err := ec3CheckFrame(render, e.numRenderChannels, e.frameLength); err != nil {
		return err
	}
	for ch, x := range render {
		bands.FloatToFloatS16Slice(x, e.render.full[ch])
	}
	e.render.splitIntoFrequencyBands()
	e.ec.AnalyzeRender(e.render.split)
	return nil
}

// ProcessCapture removes the echo from one 10 ms capture frame in place,
// capture[channel][sample] with NumCaptureChannels channels of FrameSize
// samples in [-1, 1].
func (e *EchoCanceller) ProcessCapture(capture [][]float32) error {
	if err := ec3CheckFrame(capture, e.numCaptureChannels, e.frameLength); err != nil {
		return err
	}
	var linear [][]float32
	if e.exportLinearOutput {
		// The linear output framer must see every frame.
		linear = e.linear
	}
	e.processCapture(capture, linear)
	return nil
}

// ProcessCaptureWithLinearOutput is like ProcessCapture and also writes the
// output of the linear echo canceller stage (before suppression) to
// linearOutput[channel][sample]: NumCaptureChannels channels of
// LinearOutputFrameSize samples in [-1, 1], at 16 kHz. It requires
// Filter.ExportLinearAecOutput in the config.
func (e *EchoCanceller) ProcessCaptureWithLinearOutput(capture, linearOutput [][]float32) error {
	if !e.exportLinearOutput {
		return ErrLinearOutputDisabled
	}
	if err := ec3CheckFrame(capture, e.numCaptureChannels, e.frameLength); err != nil {
		return err
	}
	if err := ec3CheckFrame(linearOutput, e.numCaptureChannels, LinearOutputFrameSize); err != nil {
		return err
	}
	e.processCapture(capture, e.linear)
	for ch, x := range e.linear {
		bands.FloatS16ToFloatSlice(x, linearOutput[ch])
	}
	return nil
}

func (e *EchoCanceller) processCapture(capture, linear [][]float32) {
	for ch, x := range capture {
		bands.FloatToFloatS16Slice(x, e.capture.full[ch])
	}
	e.ec.AnalyzeCapture(e.capture.full)
	e.capture.splitIntoFrequencyBands()
	levelChange := e.echoPathGainChange
	e.echoPathGainChange = false
	e.ec.ProcessCapture(e.capture.split, linear, levelChange)
	e.capture.mergeFrequencyBands()
	for ch, x := range e.capture.full {
		bands.FloatS16ToFloatSlice(x, capture[ch])
	}
}

// NotifyEchoPathGainChange flags a change of the echo path gain, such as an
// analog microphone gain change, for the next capture frame.
func (e *EchoCanceller) NotifyEchoPathGainChange() { e.echoPathGainChange = true }

// SetAudioBufferDelay provides an external estimate of the audio buffer
// delay in milliseconds. With Delay.UseExternalDelayEstimator set in the
// config, it is the delay the canceller aligns render and capture with.
func (e *EchoCanceller) SetAudioBufferDelay(delayMs int) { e.ec.SetAudioBufferDelay(delayMs) }

// SetCaptureOutputUsage tells the canceller whether the capture output is
// used. When it is not (for instance while the endpoint is muted), some of
// the processing is deactivated.
func (e *EchoCanceller) SetCaptureOutputUsage(used bool) { e.ec.SetCaptureOutputUsage(used) }

// Metrics returns the current echo canceller metrics.
func (e *EchoCanceller) Metrics() Metrics {
	m := e.ec.GetMetrics()
	return Metrics{
		EchoReturnLoss:            m.EchoReturnLoss,
		EchoReturnLossEnhancement: m.EchoReturnLossEnhancement,
		DelayMs:                   m.DelayMs,
	}
}

// NumRenderChannels returns the number of render channels.
func (e *EchoCanceller) NumRenderChannels() int { return e.numRenderChannels }

// NumCaptureChannels returns the number of capture channels.
func (e *EchoCanceller) NumCaptureChannels() int { return e.numCaptureChannels }
