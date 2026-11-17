# goaec3

Pure-Go acoustic echo canceller: a port of WebRTC's AEC3, aiming for bit-exact
output against the pinned libwebrtc m141, with no cgo.

goaec3 is a plain-Go AEC3. Its hot paths use Go's `simd/archsimd`
(`GOEXPERIMENT=simd`: NEON on arm64, SSE2/AVX2 on amd64) and run without
allocating once warmed up. Each lane
reproduces the floating-point operation order of the libwebrtc code path it
mirrors, and a live C++ oracle proves the match (see
[Parity & testing](#parity--testing)).

> **Status: work in progress.** The foundation (Ooura FFT, AEC3 core types,
> config) and the oracle harness are in place; the AEC3 modules are being
> ported. There is no public API yet.

## Install

```sh
go get github.com/GetStream/goaec3
```

Requires Go 1.27.4 or newer. Build with `GOEXPERIMENT=simd` to enable the SIMD
kernels; without it goaec3 runs the same lanes in pure Go, bit-exact but slower.

## Quick start

```go
import "github.com/GetStream/goaec3"

ec, err := goaec3.New(goaec3.DefaultConfig(), 48000, 1, 1) // rate, render ch, capture ch
if err != nil {
	log.Fatal(err)
}

// 10 ms frames, frame[channel][sample], samples in [-1, 1].
render := [][]float32{make([]float32, ec.FrameSize())}
capture := [][]float32{make([]float32, ec.FrameSize())}

for {
	// ... fill render with the frame sent to the loudspeaker
	// ... fill capture with the frame recorded by the microphone
	if err := ec.AnalyzeRender(render); err != nil {
		log.Fatal(err)
	}
	if err := ec.ProcessCapture(capture); err != nil { // echo removed in place
		log.Fatal(err)
	}
}
```

Frames are caller-owned; the hot path copies what it needs and allocates
nothing once warmed up. `AnalyzeRender` may run on a different goroutine than
the capture-side calls.

## Features

| Area | goaec3 |
| --- | --- |
| Reference | WebRTC m141 (`branch-heads/7390`, `d2eaa557`) |
| Sample rates | 16, 32, 48 kHz (band splitting included) |
| Channels | Multichannel render and capture |
| Configuration | Full `EchoCanceller3Config`, defaults and `Validate` matching C++ |
| SIMD | `simd/archsimd` (`GOEXPERIMENT=simd`): arm64 NEON, amd64 SSE2/AVX2; `purego` scalar fallback |
| Allocation | Zero allocations in steady-state processing |

## Public API

| API | Description |
| --- | --- |
| `type Config = aec3.EchoCanceller3Config` | Full AEC3 configuration; field names are the C++ names in CamelCase. `Config.Validate()` clamps it like `EchoCanceller3Config::Validate`. |
| `DefaultConfig() Config` | The default-constructed `EchoCanceller3Config`. |
| `MultichannelConfig() Config` | `EchoCanceller3Config::CreateDefaultMultichannelConfig()`. |
| `New(cfg, sampleRateHz, numRenderChannels, numCaptureChannels)` | Creates an `*EchoCanceller` for 16, 32 or 48 kHz. |
| `NewWithMultichannelConfig(cfg, mcCfg, ...)` | Like `New`, switching to `mcCfg` while proper multichannel render content is detected. |
| `(*EchoCanceller).AnalyzeRender(render [][]float32) error` | Buffers one 10 ms far-end frame. |
| `(*EchoCanceller).ProcessCapture(capture [][]float32) error` | Removes the echo from one 10 ms near-end frame in place. |
| `(*EchoCanceller).ProcessCaptureWithLinearOutput(capture, linear [][]float32) error` | Also returns the linear filter output (16 kHz, 160 samples per channel); needs `Filter.ExportLinearAecOutput`. |
| `(*EchoCanceller).SetAudioBufferDelay(ms int)` | External estimate of the audio buffer delay; with `Delay.UseExternalDelayEstimator` it sets the alignment. |
| `(*EchoCanceller).SetCaptureOutputUsage(used bool)` | Signals whether the output is used (e.g. muted), letting AEC3 reduce processing. |
| `(*EchoCanceller).NotifyEchoPathGainChange()` | Flags an echo path gain change (e.g. analog mic gain) for the next capture frame. |
| `(*EchoCanceller).Metrics() Metrics` | ERL, ERLE (dB) and the estimated delay (ms). |

Samples are converted to and from the 16-bit float scale AEC3 works in, and
32/48 kHz frames are split into 16 kHz bands and merged back, exactly as
`webrtc::AudioBuffer` does for the float `ProcessStream` interface; the
output matches libwebrtc driven through `AudioBuffer` bit for bit. Errors
(`ErrSampleRate`, `ErrNumChannels`, `ErrChannelCount`, `ErrFrameLength`,
`ErrLinearOutputDisabled`) are sentinel values.

## Parity & testing

The pinned reference is WebRTC m141. Sources are fetched into `tmp_check/` and
are not committed. Parity is proven against a live C++ oracle:
`internal/webrtcoracle` (build tag `webrtc_oracle`) links the prebuilt
libwebrtc through cgo.

- **Bit-exact kernels and modules.** Every ported module is compared
  bit-for-bit (`math.Float32bits` equality) against its C++ counterpart over
  many blocks of realistic input.
- **Two lanes.** The `purego` build follows the scalar C++ path. The default
  build follows the SIMD path libwebrtc selects on that architecture. WebRTC's
  SIMD variants reorder reductions, so each lane is checked against its own
  reference.
- **Floating point.** libwebrtc is built with `-ffp-contract=off`, so goaec3
  never fuses a multiply-add except where the C++ uses an explicit FMA
  intrinsic.
- **libm.** Only `exp2f`, `exp10f` and `log10f` reach libm. Platform
  libm float results are not correctly rounded and differ between Apple libm
  and glibc, so goaec3 pins them to the correctly rounded value. The oracle
  binary interposes the same functions so that the comparison covers the
  algorithm rather than the host libm.

```sh
GOEXPERIMENT=simd go test ./...                            # default lane, SIMD kernels
go test -tags purego ./...                                 # scalar lane
GOEXPERIMENT=simd go test -tags webrtc_oracle ./internal/webrtcoracle/   # C++ parity (darwin/arm64)
go test -tags 'webrtc_oracle purego' ./internal/webrtcoracle/
```

## Performance

_TODO._

## Docs

- [AGENTS.md](AGENTS.md): porting rules, lanes, float discipline, layout.

## License

_TODO._ goaec3 is derived from WebRTC (BSD 3-Clause), whose copyright and
license notice will ship with the project license.
