# goaec3

Pure-Go acoustic echo canceller: a port of WebRTC's AEC3, aiming for bit-exact
output against the pinned libwebrtc m141, with no cgo.

goaec3 is a plain-Go AEC3. Its hot paths use SIMD assembly (NEON on arm64,
SSE2/AVX2 on amd64) and run without allocating once warmed up. Each lane
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

Requires Go 1.25 or newer.

## Quick start

_TODO: the public API lands with the `EchoCanceller3` port._ The planned hot
path takes caller-owned 10 ms frames and allocates nothing:

```go
func (e *EchoCanceller) AnalyzeRender(render [][]float32) error
func (e *EchoCanceller) ProcessCapture(capture [][]float32) error
```

## Features

| Area | goaec3 |
| --- | --- |
| Reference | WebRTC m141 (`branch-heads/7390`, `d2eaa557`) |
| Sample rates | 16, 32, 48 kHz (band splitting included) |
| Channels | Multichannel render and capture |
| Configuration | Full `EchoCanceller3Config`, defaults and `Validate` matching C++ |
| SIMD | arm64 NEON, amd64 SSE2/AVX2; `purego` scalar fallback |
| Allocation | Zero allocations in steady-state processing |

## Public API

_TODO._

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
- **libm.** Only `exp2f` and `exp10f` reach libm on the signal path. Platform
  libm float results are not correctly rounded and differ between Apple libm
  and glibc, so goaec3 pins both to the correctly rounded value. The oracle
  binary interposes the same functions so that the comparison covers the
  algorithm rather than the host libm.

```sh
go test ./...                                              # default (SIMD) lane
go test -tags purego ./...                                 # scalar lane
go test -tags webrtc_oracle ./internal/webrtcoracle/       # C++ parity (darwin/arm64)
go test -tags 'webrtc_oracle purego' ./internal/webrtcoracle/
```

## Performance

_TODO._

## Docs

- [AGENTS.md](AGENTS.md): porting rules, lanes, float discipline, layout.

## License

_TODO._ goaec3 is derived from WebRTC (BSD 3-Clause), whose copyright and
license notice will ship with the project license.
