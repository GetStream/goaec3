# Agent Instructions

goaec3 is a pure-Go port of WebRTC's AEC3 echo canceller
(`modules/audio_processing/aec3`) targeting **bit-exact parity with libwebrtc
m141** (`branch-heads/7390`, commit `d2eaa5570fc9959f8dbde32912a16366b8ee75f4`).
The layout and discipline follow gopus (`../gopus/AGENTS.md`).

State current behavior in the present tense in code comments and docs. Do not
write changelog narrative.

## Prime directive: parity, proven against a live C++ oracle

- The reference sources live in `tmp_check/webrtc-m141/` (sparse checkout; not
  committed). `tmp_check/libwebrtc` is a symlink to the prebuilt
  `libwebrtc.a` (m141, darwin/arm64, NEON, built with `-ffp-contract=off`).
- `internal/webrtcoracle` (build tag `webrtc_oracle`, cgo) links that library
  and exposes C++ entry points to parity tests. Scalar (non-NEON) reference
  code that the prebuilt library does not contain is compiled from the
  checked-out sources inside the oracle package under renamed symbols, with
  `#pragma clang fp contract(off)`.
- Every ported module gets a differential test against the oracle before the
  next module builds on it. Kernels must be **bit-exact**
  (`math.Float32bits` equality). **Never weaken a gate to make work pass.**

## Lanes

- `purego` build tag: the scalar C++ path (`Aec3Optimization::kNone` and the
  scalar Ooura FFT). Bit-exact on every architecture.
- Default build: reproduces the SIMD path libwebrtc selects on that
  architecture (NEON on arm64, SSE2/AVX2 on amd64). SIMD variants in WebRTC
  reorder reductions, so the Go code for each lane follows that lane's
  operation order exactly.

## SIMD: GOEXPERIMENT=simd

- Toolchain: Go 1.27.4 (latest 1.27.x). Kernels are written in Go with
  `simd/archsimd`, which exists only when building with `GOEXPERIMENT=simd`
  and covers both arm64 (NEON) and amd64 (SSE/AVX/AVX2/AVX-512) in Go 1.27.
- Kernel file pattern:
  - `foo_simd.go`: `//go:build goexperiment.simd && (arm64 || amd64) && !purego`
    (split into `foo_simd_arm64.go` / `foo_simd_amd64.go` when the lane order
    differs per arch, which it does for NEON vs SSE2/AVX2 kernels).
  - `foo_nosimd.go`: `//go:build !goexperiment.simd || purego || !(arm64 || amd64)`
    — routes to the pure-Go lane references.
- Every lane also has a pure-Go reference (`fooNeonRef`, `fooSse2Ref`, ...).
  It is the spec, the fallback when the experiment is off, and the oracle that
  the archsimd kernel is tested against bit-exactly. A build without
  `GOEXPERIMENT=simd` is still bit-exact; only slower.
- archsimd semantics: `a.Mul(b).Add(c)` compiles to separate multiply and add
  (unfused, like `vmlaq_f32`/`_mm_mul_ps`+`_mm_add_ps`); `a.MulAdd(b, c)` is a
  real FMA (like `vfmaq_f32`/`_mm256_fmadd_ps`). Use each exactly where the C++
  does.
- Hand-written Go assembly is the exception: only for a kernel that archsimd
  cannot express (missing shuffle, etc.), with a comment naming the missing
  operation. Gate x86 paths with `archsimd.X86.AVX2()` etc.

- C++ classes that take an `Aec3Optimization` argument keep it in Go so tests
  can drive both lanes against the oracle in one binary.

## Float discipline

- All signal-domain storage and arithmetic is `float32` (C `float`). Use
  `float64` only where the C++ uses `double` (e.g. literals like
  `3.0102999566398121 * x` promote through double).
- Go may fuse `x*y + z` into an FMA on arm64. libwebrtc is built with
  contraction off and NEON `vmlaq_f32` is unfused, so wrap every product that
  feeds an add/sub in `float32(...)`: `float32(a*b) + float32(c*d)`. Use a real
  FMA only where the C++ calls `vfmaq_f32`/`_mm256_fmadd_ps`.
- Transcendentals: `sqrtf` is exact via `float32(math.Sqrt(float64(x)))`.
  AEC3's signal path only reaches libm through `exp2f` (from
  `std::pow(2.0f, x)`) and `__exp10f` (from `powf(10.f, x)`); use
  `Exp2f`/`Exp10f` in `internal/aec3/cmath.go`. The oracle binary interposes
  those same Go functions for the libm symbols, because host libm float
  results are not correctly rounded and differ between Apple libm and glibc.
  Any new libm call needs the same treatment: add a helper there and an
  interposer in `internal/webrtcoracle/libm.go`.

## Naming

- One Go file per C++ file (`matched_filter.cc` → `matched_filter.go`).
- C++ class → Go type of the same name; constructor `NewX`; methods keep their
  C++ names (exported). `kFoo` constants drop the `k` (`FftLengthBy2Plus1`).
  `std::array<float, N>` → `[N]float32`; `size_t`/`int` → `int`.
- Config: `EchoCanceller3Config` in `internal/aec3/config.go`, fields are the
  C++ snake_case names in CamelCase.
- ApmDataDumper, metrics/histogram reporting and logging are not ported.
  Field trials are fixed at their defaults (all disabled).

## Allocation discipline

Steady-state processing is allocation-free. Lock new hot paths with a
`testing.AllocsPerRun(...) == 0` test after warm-up.

## Build and test

- Use Go 1.27.4 (`$HOME/sdk/go1.27.4/bin/go`, or `go1.27.4`).
- `GOEXPERIMENT=simd go test ./...` — default lane with archsimd kernels.
- `go test ./...` — default lane, pure-Go lane references (no experiment).
- `go test -tags purego ./...` — scalar lane.
- `GOARCH=amd64 GOEXPERIMENT=simd go test ./...` — amd64 lane (Rosetta;
  AVX2 available).
- `GOEXPERIMENT=simd go test -tags webrtc_oracle ./internal/webrtcoracle/` —
  parity against libwebrtc (darwin/arm64 only; needs `tmp_check/`).

## Layout

- `internal/aec3` — the AEC3 core (one package, like the C++ directory).
- `internal/ooura` — 128-point Ooura real FFT (scalar + NEON/SSE2 lanes).
- `internal/cpufeat` — CPU feature detection (copied from gopus).
- `internal/webrtcoracle` — cgo oracle harness (tag `webrtc_oracle`).
- root package `goaec3` — public API.
