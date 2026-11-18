# goaec3

Pure-Go port of WebRTC's AEC3 acoustic echo canceller. Its output is
bit-exact with libwebrtc m141, it uses SIMD through Go's `simd/archsimd`, it
needs no cgo, and it allocates nothing in steady state.

goaec3 takes the signal played to the loudspeaker (render) and the signal
recorded by the microphone (capture), and removes the echo of the first from
the second, 10 ms at a time. Every module is ported from
`modules/audio_processing/aec3` and proven against a live C++ reference, in
the scalar lane and in the SIMD lane libwebrtc selects on each architecture
(see [Parity & testing](#parity--testing)).

> **Status: pre-v1.** The full AEC3 pipeline is ported and bit-exact against
> libwebrtc end to end on arm64 (NEON) and amd64 (SSE2, AVX2) and in the scalar
> lane. The API may still change.

## Install

```sh
go get github.com/GetStream/goaec3
```

Requires Go 1.27 or newer. Build with `GOEXPERIMENT=simd` for the SIMD
kernels; without it goaec3 runs the same lanes in pure Go, bit-exact and
slower.

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
	m := ec.Metrics() // ERL, ERLE (dB), estimated delay (ms)
	_ = m
}
```

- Feed render and capture at the same pace: one render frame per capture
  frame, with silence (zeros) when nothing plays. Long gaps in either stream
  make AEC3 lose its delay estimate.
- `AnalyzeRender` may run on a different goroutine than `ProcessCapture`,
  like WebRTC's render and capture threads. `NotifyEchoPathGainChange` may be
  called from any goroutine.
- Frames are caller-owned. Once warmed up, the hot path allocates nothing.
- Non-finite input (NaN) poisons the canceller's state permanently, exactly
  as in libwebrtc: every later output is NaN while the metrics can still look
  healthy. Sanitize input upstream, or create a new canceller.
- Default settings find echo delays up to about 0.5 s; raise
  `Config.Delay.NumFilters` for longer paths (each filter adds about 96 ms).

## Features

| Area | goaec3 |
| --- | --- |
| Reference | WebRTC m141 (`branch-heads/7390`, `d2eaa557`) |
| Sample rates | 16, 32, 48 kHz, with WebRTC's band splitting and merging |
| Channels | Multichannel render and capture, stereo detection, multichannel config |
| Configuration | The full `EchoCanceller3Config` (134 fields), defaults and `Validate` as in C++ |
| Outputs | Echo-cancelled capture, optional linear filter output, ERL/ERLE/delay metrics |
| SIMD | `simd/archsimd`: arm64 NEON, amd64 SSE2 and AVX2/FMA; pure-Go and `purego` fallbacks |
| Assembly | None |
| Allocation | Zero in steady state |

## Public API

| API | Description |
| --- | --- |
| `type Config` | The AEC3 configuration (`webrtc::EchoCanceller3Config`); field names are the C++ names in CamelCase. `Config.Validate()` clamps it like the C++. |
| `DefaultConfig()`, `MultichannelConfig()` | The default config and `CreateDefaultMultichannelConfig()`. |
| `New(cfg, sampleRateHz, numRenderChannels, numCaptureChannels)` | Creates an `*EchoCanceller` for 16, 32 or 48 kHz. |
| `NewWithMultichannelConfig(cfg, mcCfg, ...)` | Like `New`, switching to `mcCfg` while multichannel render content is detected. |
| `AnalyzeRender(render [][]float32) error` | Buffers one 10 ms render (far-end) frame. |
| `ProcessCapture(capture [][]float32) error` | Removes the echo from one 10 ms capture (near-end) frame in place. |
| `ProcessCaptureWithLinearOutput(capture, linear [][]float32) error` | Also returns the linear filter output at 16 kHz; needs `Filter.ExportLinearAecOutput`. |
| `SetAudioBufferDelay(ms int)` | External audio buffer delay; with `Delay.UseExternalDelayEstimator` it sets the alignment. |
| `SetCaptureOutputUsage(used bool)` | Tells AEC3 whether the output is used (e.g. while muted). |
| `NotifyEchoPathGainChange()` | Flags an echo path gain change (e.g. an analog mic gain change). |
| `Metrics() Metrics` | ERL and ERLE in dB, estimated delay in ms. |

`New` returns `ErrConfig` for a config that `Validate` would change, a config
libwebrtc itself cannot run (no matched filters, a refined filter shorter
than 9 blocks, a subband region reaching bin 65, negative buffering fields),
or a multichannel config that disagrees with the main one on the settings
both must share. Other errors (`ErrSampleRate`, `ErrNumChannels`,
`ErrChannelCount`, `ErrFrameLength`, `ErrLinearOutputDisabled`) are sentinel
values.

Samples are converted to and from the 16-bit float scale AEC3 works in, and
32/48 kHz frames are split into 16 kHz bands and merged back, exactly as
`webrtc::AudioBuffer` does for the float `ProcessStream` interface.

## Parity & testing

The reference is WebRTC m141. Parity is checked against two C++ builds:

- **Prebuilt libwebrtc** (`internal/webrtcoracle`, tag `webrtc_oracle`):
  the release `libwebrtc.a` for darwin/arm64, linked through cgo.
- **From source** (tag `webrtc_src_oracle`): all of AEC3 and its
  dependencies compiled from the pinned WebRTC checkout
  (`scripts/fetch-webrtc.sh`), with the arch's real SIMD selection. It runs on
  linux and darwin, arm64 and amd64, so CI checks the NEON, SSE2, AVX2 and
  scalar lanes on real hardware. On darwin/arm64 it agrees with the prebuilt
  library on every end-to-end scenario.

What is checked, every value bit for bit (`math.Float32bits` equality):

- **Every module and kernel** against its C++ counterpart, over thousands of
  blocks of realistic input, in every lane.
- **End to end**: `EchoCanceller3` driven through `webrtc::AudioBuffer` at
  16/32/48 kHz, mono and stereo, with external and fixed delays, linear
  output, stereo switching, saturation, uneven render/capture call patterns,
  and the public API's float conversion.
- **Differential fuzzing** against libwebrtc over random validated configs
  (all 134 fields), call patterns and audio, with 60 s+ runs.
- **Edge inputs**: NaN, ±Inf, ±0 and extreme gains, including C++
  `std::min`/`std::max` semantics; non-finite inputs give the same output as
  libwebrtc (NaN payloads aside).

Floating point follows the C++ build: libwebrtc is built with
`-ffp-contract=off`, so goaec3 fuses multiply-adds only where the C++ uses an
FMA intrinsic, and a test fails if the compiler fuses one anywhere else.
AEC3 reaches libm only through `exp2f`, `exp10f` and `log10f`. Their float
results differ between Apple libm and glibc, so goaec3 uses the correctly
rounded value, and the oracle substitutes the same functions. Against stock,
unmodified libwebrtc, 18 scenarios of 60 s each matched bit for bit too with
Apple libm. With glibc 2.39 the audio and linear output are bit-exact as well;
only the ERL metric differs, by at most 2.4e-6 dB, because glibc's `log10f` is
not correctly rounded.

Robustness: native fuzzing of the public API, a one-hour soak per setup
(echo reduction holds at 31–37 dB), `-race` tests for concurrent render and
capture, and boundary sweeps over every config field.

```sh
make test test-purego test-simd test-amd64   # unit tests, every lane
make fetch-webrtc oracle oracle-src          # C++ parity (prebuilt and from source)
make bench                                   # Go vs libwebrtc
```

## Performance

Time per 10 ms frame (`AnalyzeRender` + `ProcessCapture`, band splitting and
float conversion included), on identical converged echo input, against
libwebrtc m141's `EchoCanceller3` driven through `webrtc::AudioBuffer`
(release build, NEON). Apple M4 Max, Go 1.27.4, median of 6 runs:

| Case | goaec3, `GOEXPERIMENT=simd` | libwebrtc, prebuilt | libwebrtc, from source (clang `-O2`) | goaec3, pure Go |
| --- | --- | --- | --- | --- |
| 16 kHz mono | **57 µs** | 64 µs | 61–65 µs | 220 µs |
| 48 kHz mono | **64 µs** | 75 µs | 68–71 µs | 257 µs |
| 48 kHz stereo | **91 µs** | 101 µs | 94–96 µs | 378 µs |

(The from-source column is the range of two runs of the C++ reference built
by `make oracle-src`.)

On arm64 with `GOEXPERIMENT=simd`, goaec3 is 10–14% faster than the prebuilt
libwebrtc and 2–12% faster than the same C++ built from source with clang
`-O2`, while producing the same output bit for bit; that is under 1% of one
core per stream at 16 kHz. Zero allocations per frame in every case.

Scope of these numbers:

- **Build flag.** Without `GOEXPERIMENT=simd` the same lanes run as plain Go,
  about 3.5–4× slower than libwebrtc on arm64. Build with the experiment.
- **amd64.** Parity on amd64 (SSE2 and AVX2) is proven in CI on real
  hardware, but speed against libwebrtc has not yet been measured on real x86
  hardware. A build without `GOEXPERIMENT=simd` on an AVX2 machine is
  especially slow: the AVX2 lane's fused multiply-adds then run as software
  float32 FMAs.

```sh
GOEXPERIMENT=simd go test -tags webrtc_oracle -run '^$' -bench BenchmarkEc3VsLibwebrtc ./internal/webrtcoracle/
```

## Docs

- [AGENTS.md](AGENTS.md): porting rules, lanes, float discipline, layout, build and test.

## License

BSD 3-Clause; see [LICENSE](LICENSE). goaec3 is derived from WebRTC and
includes the Ooura FFT; their notices are in the same file.
