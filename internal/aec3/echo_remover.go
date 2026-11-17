package aec3

// EchoControlMetrics mirrors EchoControl::Metrics (api/audio/echo_control.h).
type EchoControlMetrics struct {
	EchoReturnLoss            float64
	EchoReturnLossEnhancement float64
	DelayMs                   int
}

// EchoRemover mirrors echo_remover.{h,cc} (the C++ interface + EchoRemoverImpl).
//
// This file is an API stub: the methods panic until the port lands.
type EchoRemover struct{}

// NewEchoRemover mirrors EchoRemover::Create.
func NewEchoRemover(config *EchoCanceller3Config, sampleRateHz, numRenderChannels, numCaptureChannels int) *EchoRemover {
	return &EchoRemover{}
}

// GetMetrics returns the current metrics.
func (e *EchoRemover) GetMetrics() EchoControlMetrics { panic("aec3: EchoRemover not ported") }

// ProcessCapture removes the echo from one capture block. externalDelay is nil
// when absent; linearOutput is nil when the linear AEC output is not exported.
func (e *EchoRemover) ProcessCapture(echoPathVariability EchoPathVariability, captureSignalSaturation bool,
	externalDelay *DelayEstimate, renderBuffer *RenderBuffer, linearOutput, capture *Block) {
	panic("aec3: EchoRemover not ported")
}

// UpdateEchoLeakageStatus reports whether echo leakage is detected.
func (e *EchoRemover) UpdateEchoLeakageStatus(leakageDetected bool) { panic("aec3: EchoRemover not ported") }

// SetCaptureOutputUsage tells the remover whether its output is used.
func (e *EchoRemover) SetCaptureOutputUsage(captureOutputUsed bool) { panic("aec3: EchoRemover not ported") }
