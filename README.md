# goaec3

Pure-Go port of WebRTC's AEC3 acoustic echo canceller, bit-exact with
libwebrtc m141.

goaec3 removes the echo of the loudspeaker signal (render) from the microphone
signal (capture), 10 ms at a time. SIMD kernels are written in Go with
`simd/archsimd`; there is no cgo, no `unsafe` and no assembly. Steady-state
processing allocates nothing, and a canceller is a handful of heap objects, so
it costs the garbage collector little. Every module is proven against a live
C++ reference (see [Parity & testing](#parity--testing)).

## Install

```sh
go get github.com/GetStream/goaec3
```

Requires Go 1.27 or newer. Build with `GOEXPERIMENT=simd` (see
[Build modes](#build-modes)).

## Quick start

```go
package main

import (
	"log"

	"github.com/GetStream/goaec3"
)

func main() {
	// Sample rate, render channels, capture channels.
	ec, err := goaec3.New(goaec3.DefaultConfig(), 48000, 1, 1)
	if err != nil {
		log.Fatal(err)
	}

	// One 10 ms frame: frame[channel][sample], samples in [-1, 1].
	render := [][]float32{make([]float32, ec.FrameSize())}
	capture := [][]float32{make([]float32, ec.FrameSize())}

	for {
		// Fill render with the frame sent to the loudspeaker and capture
		// with the frame recorded by the microphone, then:
		if err := ec.AnalyzeRender(render); err != nil {
			log.Fatal(err)
		}
		if err := ec.ProcessCapture(capture); err != nil { // echo removed in place
			log.Fatal(err)
		}
	}
}
```

## Using it correctly

- **Frames** are exactly 10 ms: `FrameSize()` (`sampleRateHz/100`) samples per
  channel, `float32` in [-1, 1], laid out `frame[channel][sample]`. Other
  shapes return `ErrChannelCount` or `ErrFrameLength`. Frames are
  caller-owned; `AnalyzeRender` copies, `ProcessCapture` writes in place.
- **Pacing**: feed one render frame per capture frame, with silence (zeros)
  when nothing plays. Long gaps in either stream make AEC3 lose its delay
  estimate.
- **Threading**: `AnalyzeRender` may run concurrently with the capture-side
  methods (`ProcessCapture`, `ProcessCaptureWithLinearOutput`,
  `SetAudioBufferDelay`, `SetCaptureOutputUsage`, `Metrics`), like WebRTC's
  render and capture threads. Calls on each side must be serialized.
  `NotifyEchoPathGainChange` may be called from any goroutine.
- **NaN**: out-of-range samples and ±Inf are clamped; NaN is not. A NaN sample
  poisons the canceller state permanently, as in libwebrtc: every later output
  is NaN while the metrics can still look healthy. Replace NaN upstream, or
  create a new canceller once one got through.
- **Delay range**: the default config finds echo delays up to about 0.5 s.
  Raise `Config.Delay.NumFilters` for longer paths; each filter adds about
  96 ms.
- **Allocation**: `AnalyzeRender` and `ProcessCapture` allocate nothing in
  steady state. A render burst deeper than any before grows the render queue
  once.

## Build modes

Every mode is memory-safe and bit-exact with the libwebrtc lane it
reproduces. They differ in speed and in which lane runs.

| Build | Lane | Kernels |
| --- | --- | --- |
| `GOEXPERIMENT=simd go build` (recommended) | The SIMD lane libwebrtc selects: NEON on arm64, AVX2 or SSE2 on amd64 | `simd/archsimd` |
| `go build` | The same lane | Pure Go, slower |
| `go build -tags purego` | The scalar lane (`Aec3Optimization::kNone`), on every architecture | Pure Go |

WebRTC's SIMD code reorders floating-point reductions, so each lane has its
own bit pattern; goaec3 follows the operation order of the lane it runs. On
amd64 the AVX2 lane is chosen under libwebrtc's condition (AVX, AVX2, BMI2 and
OS support for YMM state), with and without the experiment.

## Features

| Area | goaec3 |
| --- | --- |
| Reference | WebRTC m141 (`branch-heads/7390`, `d2eaa557`) |
| Sample rates | 16, 32, 48 kHz, with WebRTC's band splitting and merging |
| Channels | Multichannel render and capture, stereo content detection, multichannel config |
| Configuration | The full `EchoCanceller3Config` (134 fields), defaults and `Validate` as in C++ |
| Outputs | Echo-cancelled capture, optional linear filter output, ERL/ERLE/delay metrics |
| SIMD | arm64 NEON, amd64 SSE2 and AVX2/FMA |
| Safety | No cgo, no `unsafe`, no assembly (`TestNoUnsafe` scans every file) |
| Allocation | Zero in steady state (`TestSteadyStateAllocations`) |
| Not ported | `ApmDataDumper`, histogram reporting, logging; field trials are fixed at their defaults |

## Public API

One package, `github.com/GetStream/goaec3`; everything else is under
`internal/`.

| API | Description |
| --- | --- |
| `type Config` | The AEC3 configuration (`webrtc::EchoCanceller3Config`); fields are the C++ names in CamelCase. `Validate()` clamps it like the C++ and reports whether it was already valid. |
| `DefaultConfig() Config` | The default-constructed config. |
| `MultichannelConfig() Config` | `CreateDefaultMultichannelConfig()`. |
| `New(cfg, sampleRateHz, numRenderChannels, numCaptureChannels) (*EchoCanceller, error)` | Creates a canceller for 16, 32 or 48 kHz. |
| `NewWithMultichannelConfig(cfg, multichannelCfg, ...)` | Like `New`; switches to `multichannelCfg` while multichannel render content is detected. |
| `AnalyzeRender(render [][]float32) error` | Buffers one 10 ms render (far-end) frame. |
| `ProcessCapture(capture [][]float32) error` | Removes the echo from one 10 ms capture (near-end) frame in place. |
| `ProcessCaptureWithLinearOutput(capture, linearOutput [][]float32) error` | Also writes the linear filter output: `LinearOutputFrameSize` samples per channel at 16 kHz. Needs `Filter.ExportLinearAecOutput`. |
| `SetAudioBufferDelay(delayMs int)` | External audio buffer delay; with `Delay.UseExternalDelayEstimator` it sets the alignment. |
| `SetCaptureOutputUsage(used bool)` | Tells AEC3 whether the output is used (for instance while muted). |
| `NotifyEchoPathGainChange()` | Flags an echo path gain change, such as an analog microphone gain change. |
| `Metrics() Metrics` | `EchoReturnLoss` and `EchoReturnLossEnhancement` in dB, `DelayMs`. |
| `SampleRateHz()`, `FrameSize()`, `NumRenderChannels()`, `NumCaptureChannels()` | The canceller's format. |

Errors are sentinel values, matched with `errors.Is`:

| Error | Returned when |
| --- | --- |
| `ErrSampleRate` | The sample rate is not 16000, 32000 or 48000 Hz. |
| `ErrNumChannels` | A channel count is below one. |
| `ErrConfig` | `Validate` would change the config, libwebrtc cannot run it (no matched filters, a refined filter shorter than 9 blocks, a subband nearend region reaching bin 65, negative buffering fields), or the two configs of `NewWithMultichannelConfig` disagree on a setting both must share. |
| `ErrChannelCount`, `ErrFrameLength` | A frame has the wrong number of channels, or a channel is not one 10 ms frame. |
| `ErrLinearOutputDisabled` | `ProcessCaptureWithLinearOutput` is called without `Filter.ExportLinearAecOutput`. |

Samples are converted to and from the 16-bit float scale AEC3 works in, and
32/48 kHz frames are split into 16 kHz bands and merged back, exactly as
`webrtc::AudioBuffer` does for the float `ProcessStream` interface.

## Parity & testing

Every comparison is bit for bit (`math.Float32bits` equality) against C++
built with `-ffp-contract=off`, in two builds:

| Oracle | Build tag | What it is |
| --- | --- | --- |
| Prebuilt | `webrtc_oracle` | The m141 `libwebrtc.a` for darwin/arm64, linked through cgo. |
| From source | `webrtc_src_oracle` | AEC3 and its dependencies compiled from the pinned WebRTC checkout with clang, in the machine's lane. Runs on linux and darwin, arm64 and amd64. |

What is checked:

- **Modules and kernels** against their C++ counterparts, in every lane:
  NEON, SSE2, AVX2 and scalar. CI runs each lane on real hardware and fails if
  either side runs a different lane than expected.
- **End to end**: `EchoCanceller3` driven through `webrtc::AudioBuffer` at
  16, 32 and 48 kHz, mono and stereo, with linear output, quiet render and
  the public API's float conversion.
- **Differential fuzzing** (`FuzzEc3Differential`,
  `TestEc3DifferentialRandomConfigs`): random valid configs over all 134
  fields, call patterns and audio, compared with libwebrtc frame by frame.
- **Edge inputs**: NaN input and C++ `std::min`/`std::max` NaN semantics give
  the same output as libwebrtc.
- **Floating point**: multiply-adds are fused only where the C++ calls an FMA
  intrinsic; `TestNoUnintendedFMA` reads the compiler's arm64 output and
  fails on any other fused multiply-add.
- **libm**: AEC3 reaches libm only through `exp2f`, `exp10f` and `log10f`,
  whose results differ between Apple libm and glibc. goaec3 uses the correctly
  rounded value and the oracle substitutes the same functions. Against
  libwebrtc on stock Apple libm, 18 scenarios of 60 s each are bit-exact in
  output, linear output and metrics. On glibc the audio is bit-exact and the
  ERL metric differs by about 1e-6 dB.

Robustness tests cover native fuzzing of the public API (`FuzzEchoCanceller`),
a soak run (`GOAEC3_SOAK_SECONDS`), concurrent render and capture under
`-race`, bad frame shapes and boundary sweeps over the config.

| Target | Runs |
| --- | --- |
| `make test` | Unit tests, default lane in pure Go |
| `make test-simd` | Unit tests, default lane with `GOEXPERIMENT=simd` |
| `make test-purego` | Unit tests, scalar lane |
| `make test-amd64` | Unit tests for amd64 (Rosetta on Apple silicon), with and without the experiment |
| `make test-race` | `-race -short`, with and without the experiment |
| `make lint` | gofmt, `go vet` in every build configuration, generated oracle files |
| `make fetch-webrtc` | Sparse checkout of the pinned WebRTC sources into `tmp_check/webrtc-m141` |
| `make oracle` | Parity against the prebuilt libwebrtc (darwin/arm64, needs `tmp_check/libwebrtc`) |
| `make oracle-src`, `oracle-src-scalar` | Parity against the source build in the machine's lane, and in the scalar lane (needs clang) |
| `make oracle-src-rosetta`, `oracle-src-rosetta-avx2` | The x86 SSE2, scalar and AVX2 lanes on Apple silicon |
| `make oracle-x86-kernels` | x86 FFT, spectrum and vector-math kernels against their C++ sources |
| `make oracle-stock`, `oracle-src-stock` | Parity on the host libm, without the libm substitution |
| `make bench` | goaec3 against libwebrtc, time per frame |

CI (`.github/workflows/ci.yml`, `macos.yml`) runs lint, the unit tests in all
three build modes on linux/amd64, linux/arm64 and macOS, the race tests, and
the source oracle in the AVX2, SSE2, NEON and scalar lanes.

## Performance

Time per 10 ms frame (`AnalyzeRender` + `ProcessCapture`, band splitting and
float conversion included) against libwebrtc m141's `EchoCanceller3` driven
through `webrtc::AudioBuffer`, on the same input, producing the same output.
goaec3 is built with `GOEXPERIMENT=simd` unless noted. Go/C++ is the time
ratio; below 1 goaec3 is faster.

| Machine, lane | 16 kHz mono | 48 kHz mono | 48 kHz stereo | Go/C++ |
| --- | --- | --- | --- | --- |
| Apple M4 Max, NEON: goaec3 | 43.3 µs | 48.8 µs | 72.8 µs | 0.65–0.72× |
| Apple M4 Max, NEON: libwebrtc | 66.8 µs | 74.5 µs | 101.3 µs | 1 |
| Apple M4 Max, NEON: goaec3 without the experiment | 91.7 µs | 101.7 µs | 162.9 µs | 1.4–1.6× |
| Neoverse-N2, NEON | | | | 0.86–0.89× |
| AMD EPYC Zen 4, AVX2 | | | | 0.78–0.86× |
| AMD EPYC Zen 3, AVX2 | | | | 0.81–0.95× |
| Intel Xeon 6975P-C, AVX2 | | | | 0.93–1.01× |
| AMD EPYC Zen 4, SSE2 | | | | 0.85–0.93× |
| AMD EPYC Zen 3, SSE2 | | | | 0.88–0.91× |

The M4 rows compare with the prebuilt libwebrtc. The other rows are
GitHub-hosted runners (`.github/workflows/bench.yml`), where libwebrtc is
built from source with clang `-O2` in the same job; the runner CPU varies per
job, so those rows give ratios only. The Intel row predates the current x86
kernels.

Without `GOEXPERIMENT=simd` the same lanes run as pure Go: about 2.0–2.3× the
time of libwebrtc on Neoverse-N2, 1.8–2.3× in the SSE2 lane and 4–5× in the
AVX2 lane, whose fused multiply-adds then run in software.

Memory and garbage collection, per canceller with the default config:

| | 16 kHz mono | 48 kHz mono | 48 kHz stereo |
| --- | --- | --- | --- |
| Live heap | 319 kB | 436 kB | 501 kB |
| Heap objects | 13 | 15 | 15 |
| Pointer slots | 60 | 64 | 77 |

Sample buffers are pointer-free arrays, so the collector follows few pointers:
one GC cycle with 1000 live cancellers takes about 1.2–1.5 ms on the M4 Max
on one CPU.

```sh
make bench                                                   # goaec3 vs libwebrtc
go test -run 'TestNewFootprint|TestPointerSlots' -v .        # memory per canceller
go test -run '^$' -bench BenchmarkGCMark -cpu 1 .            # GC cycle, 1000 cancellers
```

## Docs

- [AGENTS.md](AGENTS.md): porting rules, lanes, SIMD and float discipline,
  layout, build and test.
- [example_test.go](example_test.go): the runnable example.

## License

BSD 3-Clause; see [LICENSE](LICENSE). goaec3 is derived from WebRTC and
includes the Ooura FFT; their notices are in the same file.
