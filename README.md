# goaec3

Pure-Go port of WebRTC's AEC3 echo canceller, bit-exact with libwebrtc m141.
SIMD through `simd/archsimd`; no cgo, no `unsafe`, no assembly; zero
allocations per frame. Faster than libwebrtc on arm64 and amd64.

> **Status: pre-v1.** The full AEC3 pipeline is ported and bit-exact; the API
> may still change.

## Install

```sh
go get github.com/GetStream/goaec3
```

Go 1.27+. Build with `GOEXPERIMENT=simd` for the SIMD kernels; without it the
same code runs in pure Go, bit-exact and slower.

## Quick start

```go
ec, err := goaec3.New(goaec3.DefaultConfig(), 48000, 1, 1) // rate, render ch, capture ch
if err != nil {
	log.Fatal(err)
}
render := [][]float32{make([]float32, ec.FrameSize())}  // [channel][sample], 10 ms, [-1, 1]
capture := [][]float32{make([]float32, ec.FrameSize())}

for {
	// fill render (sent to the loudspeaker) and capture (from the microphone)
	if err := ec.AnalyzeRender(render); err != nil {
		log.Fatal(err)
	}
	if err := ec.ProcessCapture(capture); err != nil { // echo removed in place
		log.Fatal(err)
	}
}
```

- Feed one render frame per capture frame, zeros when nothing plays.
- `AnalyzeRender` may run on another goroutine than `ProcessCapture`.
- A NaN sample poisons the state permanently, as in libwebrtc: sanitize input.
- Default settings find echo delays up to ~0.5 s; raise
  `Config.Delay.NumFilters` for longer paths (~96 ms each).

## API

| | |
| --- | --- |
| `New(cfg, rateHz, renderCh, captureCh)` | 16, 32 or 48 kHz; `NewWithMultichannelConfig` adds a stereo config |
| `AnalyzeRender`, `ProcessCapture` | One 10 ms frame each |
| `ProcessCaptureWithLinearOutput` | Also returns the linear filter output (16 kHz) |
| `SetAudioBufferDelay`, `SetCaptureOutputUsage`, `NotifyEchoPathGainChange` | As in WebRTC |
| `Metrics()` | ERL, ERLE (dB), delay (ms) |
| `Config`, `DefaultConfig()`, `MultichannelConfig()` | The full `EchoCanceller3Config` |

Errors are sentinels (`ErrConfig`, `ErrSampleRate`, `ErrFrameLength`, ...);
`New` rejects configs that `Validate` would change or libwebrtc cannot run.

## Performance

![goaec3 time per 10 ms frame as a percentage of libwebrtc](docs/perf.svg)

Band splitting and float conversion included; on Apple M4 Max that is
43.3 / 48.8 / 72.8 µs against libwebrtc's 66.8 / 74.5 / 101.3 µs. M4 compares
with the prebuilt libwebrtc, the others run on GitHub runners against libwebrtc
built from source (`.github/workflows/bench.yml`). With a silent or quiet far
end goaec3 skips provably redundant work and runs at 0.53–0.70×. Pure Go (no
experiment) is 1.4–1.6× on M4, 1.9–2.3× on Neoverse-N2 and SSE2.

A canceller is 13–15 heap objects (~320–500 kB); 1000 live cancellers cost
about 1.3 ms per GC cycle.

## Parity

Every module, kernel and the whole pipeline are checked bit for bit against
libwebrtc (prebuilt, and built from the pinned sources) in every lane — NEON,
SSE2, AVX2, scalar — on real hardware in CI, plus differential fuzzing over
random configs and edge inputs. See [AGENTS.md](AGENTS.md).

```sh
make test test-simd test-purego   # unit tests
make fetch-webrtc oracle-src      # parity against libwebrtc built from source
make bench                        # goaec3 vs libwebrtc
```

## License

BSD 3-Clause, see [LICENSE](LICENSE). Derived from WebRTC; includes the Ooura
FFT.
