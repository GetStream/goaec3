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
// AnalyzeRender and ProcessCapture allocate nothing in steady state; a render
// burst deeper than any before grows the queue once.
//
// # Untrusted input
//
// No frame makes a method panic: a frame of the wrong shape is an error that
// leaves the canceller unchanged, and every float32 value is accepted. A NaN
// sample is not an error but breaks the canceller for good (see
// [EchoCanceller.ProcessCapture]).
//
// New rejects every config it cannot run, and bounds the memory and the work
// per frame of the ones it accepts: at most [MaxChannels] channels a side,
// filters of at most [MaxFilterLengthBlocks] blocks and at most
// [MaxDelayFilters] delay estimation filters, next to the ranges of
// Config.Validate. The largest canceller New creates takes less than 64 MB
// and on the order of a hundred times the work per frame of the default one:
// a service that takes configs from its users sets its own, lower limits.
//
// Breaking the concurrency contract of [EchoCanceller] is a data race. It is
// not detected: the output is undefined and a method may panic.
package goaec3

import (
	"errors"
	"fmt"
	"sync/atomic"

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

// Limits on what New accepts. They bound the memory of a canceller and its
// work per frame, both of which grow with the product of the channel counts
// and the filter lengths, and are far above what an echo canceller uses.
const (
	// MaxChannels is the largest number of render and of capture channels.
	// The defaults of libwebrtc are tuned for mono and stereo.
	MaxChannels = 16
	// MaxFilterLengthBlocks is the largest Filter.Refined.LengthBlocks and
	// Filter.Coarse.LengthBlocks: an echo tail of one second. The default is
	// 13 blocks. Config.Validate sets no upper limit.
	MaxFilterLengthBlocks = 250
	// MaxDelayFilters is the largest Delay.NumFilters: a delay search range
	// of six seconds or more. The default is 5. Config.Validate allows 5000,
	// which makes the render buffer hundreds of megabytes per channel.
	MaxDelayFilters = 64
)

var (
	// ErrSampleRate is returned by New for a sample rate other than 16000,
	// 32000 or 48000 Hz.
	ErrSampleRate = errors.New("goaec3: sample rate must be 16000, 32000 or 48000 Hz")
	// ErrNumChannels is returned by New for a channel count below one or
	// above MaxChannels.
	ErrNumChannels = errors.New("goaec3: channel counts must be between 1 and 16")
	// ErrChannelCount is returned when a frame has the wrong number of
	// channels.
	ErrChannelCount = errors.New("goaec3: frame has the wrong number of channels")
	// ErrFrameLength is returned when a channel does not hold exactly one
	// 10 ms frame.
	ErrFrameLength = errors.New("goaec3: channel does not hold one 10 ms frame")
	// ErrConfig is returned by New and NewWithMultichannelConfig for a config
	// that is outside its legal ranges (Config.Validate would change it), that
	// libwebrtc itself cannot run, that exceeds MaxFilterLengthBlocks or
	// MaxDelayFilters, or, for NewWithMultichannelConfig, a pair of configs
	// that disagree on the settings both must share.
	ErrConfig = errors.New("goaec3: invalid config")
	// ErrLinearOutputDisabled is returned by ProcessCaptureWithLinearOutput
	// when Filter.ExportLinearAecOutput is not set in the config.
	ErrLinearOutputDisabled = errors.New("goaec3: linear output requires Filter.ExportLinearAecOutput")
)

// ec3Side converts one signal direction between the caller's full-band
// frames and the split-band frame of the canceller. At 32 and 48 kHz it
// keeps a FloatS16 copy of the frame in the canceller's extra samples, from
// fullOffset, and uses one band-splitting filter per channel, from
// firstSplitter. At 16 kHz the single band is the full-band signal.
// ec3Side holds no pointers.
type ec3Side struct {
	numChannels   int
	numBands      int
	frameLength   int
	fullOffset    int
	firstSplitter int
}

// init sets up s and returns the number of extra samples it uses.
func (s *ec3Side) init(sampleRateHz, numChannels, fullOffset, firstSplitter int) int {
	s.numChannels = numChannels
	s.frameLength = sampleRateHz / 100
	s.numBands = s.frameLength / aec3.SplitBandSize
	s.fullOffset, s.firstSplitter = fullOffset, firstSplitter
	if s.numBands > 1 {
		return numChannels * s.frameLength
	}
	return 0
}

// full returns the full-band FloatS16 samples of every channel.
func (s *ec3Side) full(extra []float32) []float32 {
	n := s.numChannels * s.frameLength
	return extra[s.fullOffset : s.fullOffset+n : s.fullOffset+n]
}

// channel returns the full-band FloatS16 samples of one channel.
func (s *ec3Side) channel(extra []float32, ch int) []float32 {
	i, n := s.fullOffset+ch*s.frameLength, s.frameLength
	return extra[i : i+n : i+n]
}

// split converts frame to FloatS16 and splits it into the bands of dst.
func (s *ec3Side) split(frame [][]float32, dst aec3.Frame, extra []float32, splitters []bands.ChannelSplitter) {
	if s.numBands == 1 {
		for ch, x := range frame {
			bands.FloatToFloatS16Slice(x, dst.View(0, ch))
		}
		return
	}
	for ch, x := range frame {
		bands.FloatToFloatS16Slice(x, s.channel(extra, ch))
	}
	s.analysis(dst, extra, splitters)
}

// analysis splits the full-band copy into the bands of dst.
func (s *ec3Side) analysis(dst aec3.Frame, extra []float32, splitters []bands.ChannelSplitter) {
	for ch := 0; ch < s.numChannels; ch++ {
		var b2 []float32
		if s.numBands == 3 {
			b2 = dst.View(2, ch)
		}
		splitters[s.firstSplitter+ch].Analysis(s.numBands, s.channel(extra, ch), dst.View(0, ch), dst.View(1, ch), b2)
	}
}

// merge merges the bands of src and converts the result to frame.
func (s *ec3Side) merge(src aec3.Frame, frame [][]float32, extra []float32, splitters []bands.ChannelSplitter) {
	if s.numBands == 1 {
		for ch, x := range frame {
			bands.FloatS16ToFloatSlice(src.View(0, ch), x)
		}
		return
	}
	for ch, x := range frame {
		var b2 []float32
		if s.numBands == 3 {
			b2 = src.View(2, ch)
		}
		full := s.channel(extra, ch)
		splitters[s.firstSplitter+ch].Synthesis(s.numBands, src.View(0, ch), src.View(1, ch), b2, full)
		bands.FloatS16ToFloatSlice(full, x)
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
// capture stream. Create one with New; the zero value is not usable.
//
// AnalyzeRender may be called concurrently with the capture-side methods
// (ProcessCapture, ProcessCaptureWithLinearOutput, SetAudioBufferDelay,
// SetCaptureOutputUsage, Metrics). Calls on each side must be serialized;
// NotifyEchoPathGainChange and the accessors may be called from any
// goroutine. Two concurrent calls on one side are a data race: nothing
// detects it, the output is undefined and a call may panic.
//
// An EchoCanceller is two heap objects: itself, whose pointer words come
// first, and a pointer-free arena that holds its arrays.
type EchoCanceller struct {
	core    aec3.Ec3Core
	render  ec3Side
	capture ec3Side

	sampleRateHz       int
	frameLength        int
	numRenderChannels  int
	numCaptureChannels int
	exportLinearOutput bool
	echoPathGainChange atomic.Bool
}

// New creates an echo canceller for sampleRateHz (16000, 32000 or 48000)
// with the given number of render and capture channels. The config must be
// valid: New returns ErrConfig if Config.Validate would change it (call
// Validate first to clamp a config into its legal ranges), if libwebrtc
// cannot run it (which includes a refined filter ErrorFloor or NoiseGate
// below 1e-12: libwebrtc then turns silence into NaN), or if it exceeds
// MaxFilterLengthBlocks or MaxDelayFilters, which Validate does not clamp
// to. New returns ErrNumChannels for more than
// MaxChannels channels a side.
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
	if numRenderChannels < 1 || numCaptureChannels < 1 ||
		numRenderChannels > MaxChannels || numCaptureChannels > MaxChannels {
		return nil, ErrNumChannels
	}
	if err := ec3CheckConfig(cfg); err != nil {
		return nil, err
	}
	if multichannelCfg != nil {
		if err := ec3CheckConfig(multichannelCfg); err != nil {
			return nil, fmt.Errorf("multichannel config: %w", err)
		}
		if err := ec3CheckCompatible(cfg, multichannelCfg); err != nil {
			return nil, err
		}
	}
	e := &EchoCanceller{
		sampleRateHz:       sampleRateHz,
		frameLength:        sampleRateHz / 100,
		numRenderChannels:  numRenderChannels,
		numCaptureChannels: numCaptureChannels,
		exportLinearOutput: cfg.Filter.ExportLinearAecOutput,
	}
	// The full-band copies are extra samples of the core: capture first,
	// then render, which AnalyzeRender writes.
	extra := e.capture.init(sampleRateHz, numCaptureChannels, 0, 0)
	extra += e.render.init(sampleRateHz, numRenderChannels, extra, numCaptureChannels)
	numSplitters := 0
	if sampleRateHz > 16000 {
		numSplitters = numCaptureChannels + numRenderChannels
	}
	e.core.Init(cfg, multichannelCfg, sampleRateHz, numRenderChannels, numCaptureChannels, extra,
		func(n aec3.ArenaSizes) aec3.Arena { return newEcArena(n, numSplitters) })
	return e, nil
}

// SampleRateHz returns the sample rate.
func (e *EchoCanceller) SampleRateHz() int { return e.sampleRateHz }

// FrameSize returns the number of samples per channel in a 10 ms frame.
func (e *EchoCanceller) FrameSize() int { return e.frameLength }

// AnalyzeRender buffers one 10 ms frame of the render signal,
// render[channel][sample] with NumRenderChannels channels of FrameSize
// samples in [-1, 1]. The frame is copied; the caller keeps ownership.
//
// A NaN sample poisons the canceller state permanently, as in libwebrtc:
// every later output is NaN while the metrics can still look healthy.
// Out-of-range values and ±Inf are clamped, NaN is not. Replace NaN
// upstream, or create a new EchoCanceller once one got through.
func (e *EchoCanceller) AnalyzeRender(render [][]float32) error {
	if err := ec3CheckFrame(render, e.numRenderChannels, e.frameLength); err != nil {
		return err
	}
	var ec aec3.EchoCanceller3
	e.core.BindRender(&ec)
	e.render.split(render, ec.RenderFrame(), ec.ExtraSamples(), e.splitters())
	ec.AnalyzeRenderFrame()
	return nil
}

// ProcessCapture removes the echo from one 10 ms capture frame in place,
// capture[channel][sample] with NumCaptureChannels channels of FrameSize
// samples in [-1, 1]. A call that returns an error has changed neither the
// frame nor the canceller.
//
// A NaN sample poisons the canceller state permanently, as in libwebrtc:
// every later output is NaN while the metrics can still look healthy.
// Out-of-range values and ±Inf are clamped, NaN is not. Replace NaN
// upstream, or create a new EchoCanceller once one got through.
func (e *EchoCanceller) ProcessCapture(capture [][]float32) error {
	if err := ec3CheckFrame(capture, e.numCaptureChannels, e.frameLength); err != nil {
		return err
	}
	var ec aec3.EchoCanceller3
	e.core.Bind(&ec)
	e.processCapture(&ec, capture)
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
	var ec aec3.EchoCanceller3
	e.core.Bind(&ec)
	e.processCapture(&ec, capture)
	linear := ec.LinearOutputFrame()
	for ch, x := range linearOutput {
		bands.FloatS16ToFloatSlice(linear.View(0, ch), x)
	}
	return nil
}

// ec3CheckConfig rejects configs that Validate would change, and the valid
// configs libwebrtc cannot run: no matched filters (it dereferences an empty
// filter bank), a refined filter shorter than 9 blocks (the reverb decay
// estimator sizes a vector with length-9 < 0) and a subband nearend region
// reaching bin 65 (read past the end of the 65-bin spectrum). Negative
// Buffering fields have no C++ equivalent (size_t there) and are rejected.
// The step size of the refined filter is h / (0.5*h*X2 + n*E2) for a render
// power X2 at or above NoiseGate and an error estimate h at or above
// ErrorFloor. Under digital silence in the capture signal E2 is 0, and with
// a floor or a gate of 0, or a product of the two that underflows, the step
// size is 0/0 or infinite and every later output is NaN, in libwebrtc too.
// Both fields are at least minRefinedStepOperand.
// It also rejects the configs over MaxFilterLengthBlocks and
// MaxDelayFilters, the fields that size the buffers and that Validate
// leaves unbounded or bounds too high. The initial filter lengths and
// Erle.NumSections are at most the full lengths in a valid config.
func ec3CheckConfig(cfg *Config) error {
	v := *cfg
	if !v.Validate() {
		return fmt.Errorf("%w: outside the legal ranges (use Config.Validate to clamp it)", ErrConfig)
	}
	switch {
	case cfg.Delay.NumFilters == 0:
		return fmt.Errorf("%w: Delay.NumFilters must be at least 1", ErrConfig)
	case cfg.Filter.Refined.LengthBlocks < 9:
		return fmt.Errorf("%w: Filter.Refined.LengthBlocks must be at least 9", ErrConfig)
	case cfg.Suppressor.UseSubbandNearendDetection &&
		(cfg.Suppressor.SubbandNearendDetection.Subband1.High >= aec3.FftLengthBy2Plus1 ||
			cfg.Suppressor.SubbandNearendDetection.Subband2.High >= aec3.FftLengthBy2Plus1):
		return fmt.Errorf("%w: SubbandNearendDetection regions must end below bin %d", ErrConfig, aec3.FftLengthBy2Plus1)
	case cfg.Buffering.ExcessRenderDetectionIntervalBlocks < 0 || cfg.Buffering.MaxAllowedExcessRenderBlocks < 0:
		return fmt.Errorf("%w: Buffering fields must not be negative", ErrConfig)
	case ec3RefinedStepUndefined(&cfg.Filter.Refined) || ec3RefinedStepUndefined(&cfg.Filter.RefinedInitial):
		return fmt.Errorf("%w: the refined filter ErrorFloor and NoiseGate must be at least %g", ErrConfig,
			minRefinedStepOperand)
	case cfg.Filter.Refined.LengthBlocks > MaxFilterLengthBlocks || cfg.Filter.Coarse.LengthBlocks > MaxFilterLengthBlocks:
		return fmt.Errorf("%w: filter LengthBlocks must be at most %d", ErrConfig, MaxFilterLengthBlocks)
	case cfg.Delay.NumFilters > MaxDelayFilters:
		return fmt.Errorf("%w: Delay.NumFilters must be at most %d", ErrConfig, MaxDelayFilters)
	}
	return nil
}

// minRefinedStepOperand is the smallest refined filter ErrorFloor and
// NoiseGate New accepts: their product is far from underflow. The defaults
// are 0.001 and 20075344.
const minRefinedStepOperand = 1e-12

func ec3RefinedStepUndefined(r *aec3.RefinedConfiguration) bool {
	return r.ErrorFloor < minRefinedStepOperand || r.NoiseGate < minRefinedStepOperand
}

// ec3CheckCompatible mirrors CompatibleConfigs in config_selector.cc: the
// settings fixed at construction must be equal in both configs.
func ec3CheckCompatible(a, b *Config) error {
	if a.Delay.FixedCaptureDelaySamples != b.Delay.FixedCaptureDelaySamples ||
		a.Filter.ExportLinearAecOutput != b.Filter.ExportLinearAecOutput ||
		a.Filter.HighPassFilterEchoReference != b.Filter.HighPassFilterEchoReference ||
		a.MultiChannel.DetectStereoContent != b.MultiChannel.DetectStereoContent ||
		a.MultiChannel.StereoDetectionTimeoutThresholdSeconds != b.MultiChannel.StereoDetectionTimeoutThresholdSeconds {
		return fmt.Errorf("%w: the configs must agree on Delay.FixedCaptureDelaySamples, Filter.ExportLinearAecOutput, "+
			"Filter.HighPassFilterEchoReference, MultiChannel.DetectStereoContent and "+
			"MultiChannel.StereoDetectionTimeoutThresholdSeconds", ErrConfig)
	}
	return nil
}

func (e *EchoCanceller) processCapture(ec *aec3.EchoCanceller3, capture [][]float32) {
	frame := ec.CaptureFrame()
	extra := ec.ExtraSamples()
	var splitters []bands.ChannelSplitter
	if e.capture.numBands > 1 {
		splitters = e.splitters()
	}
	if e.capture.numBands == 1 {
		for ch, x := range capture {
			bands.FloatToFloatS16Slice(x, frame.View(0, ch))
		}
		ec.AnalyzeCaptureSamples(frame.Samples())
	} else {
		for ch, x := range capture {
			bands.FloatToFloatS16Slice(x, e.capture.channel(extra, ch))
		}
		ec.AnalyzeCaptureSamples(e.capture.full(extra))
		e.capture.analysis(frame, extra, splitters)
	}
	levelChange := e.echoPathGainChange.Swap(false)
	// The linear output framer must see every frame.
	ec.ProcessCaptureFrame(e.exportLinearOutput, levelChange)
	e.capture.merge(frame, capture, extra, splitters)
}

// NotifyEchoPathGainChange flags a change of the echo path gain, such as an
// analog microphone gain change, for the next capture frame. It may be called
// from any goroutine.
func (e *EchoCanceller) NotifyEchoPathGainChange() { e.echoPathGainChange.Store(true) }

// SetAudioBufferDelay provides an external estimate of the audio buffer
// delay in milliseconds. With Delay.UseExternalDelayEstimator set in the
// config, it is the delay the canceller aligns render and capture with.
func (e *EchoCanceller) SetAudioBufferDelay(delayMs int) {
	var ec aec3.EchoCanceller3
	e.core.Bind(&ec)
	ec.SetAudioBufferDelay(delayMs)
}

// SetCaptureOutputUsage tells the canceller whether the capture output is
// used. When it is not (for instance while the endpoint is muted), some of
// the processing is deactivated.
func (e *EchoCanceller) SetCaptureOutputUsage(used bool) {
	var ec aec3.EchoCanceller3
	e.core.Bind(&ec)
	ec.SetCaptureOutputUsage(used)
}

// Metrics returns the current echo canceller metrics.
func (e *EchoCanceller) Metrics() Metrics {
	var ec aec3.EchoCanceller3
	e.core.Bind(&ec)
	m := ec.GetMetrics()
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

// splitters returns the band-splitting filters of e, which its arena holds.
func (e *EchoCanceller) splitters() []bands.ChannelSplitter {
	return e.core.Arena().(ecArena).splitters()
}
