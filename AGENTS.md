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
- No hand-written assembly. Kernels are archsimd; a kernel archsimd cannot
  express stays pure Go. Gate x86 kernels with `archsimd.X86.AVX2()` etc.
- End every 256-bit archsimd section with `archsimd.ClearAVXUpperBits()`
  (VZEROUPPER). Go's scalar float code is legacy-SSE encoded, so a dirty upper
  YMM state makes every later scalar float op pay the AVX→SSE transition
  penalty: on Intel this made the AVX2 lane 16× slower per frame.
- CPU detection (`DetectOptimization`): one routine for builds with and
  without `GOEXPERIMENT=simd`, `golang.org/x/sys/cpu` with libwebrtc's kAVX2
  condition (AVX, OSXSAVE/YMM, AVX2, BMI2), so both builds pick the same lane
  (the lane order still depends on AVX2 when the lane runs as pure Go). arm64
  is always NEON; purego is always scalar. The library never reads the
  environment: tests force a lane with `aec3.SetForcedLaneForTesting`
  (internal, unreachable by library users); the source oracle's TestMain
  applies `GOAEC3_FORCE_LANE` through it.

- C++ classes that take an `Aec3Optimization` argument keep it in Go so tests
  can drive both lanes against the oracle in one binary.

## Float discipline

- All signal-domain storage and arithmetic is `float32` (C `float`). Use
  `float64` only where the C++ uses `double` (e.g. literals like
  `3.0102999566398121 * x` promote through double).
- Go may fuse `x*y + z` into an FMA on arm64, also across statements. libwebrtc
  is built with contraction off and NEON `vmlaq_f32` is unfused, so wrap every
  product that feeds an add/sub in `float32(...)`: `float32(a*b) + float32(c*d)`,
  including products stored in locals (`a := float32(x * y)` before `1 - a`). Use a real
  FMA only where the C++ calls `vfmaq_f32`/`_mm256_fmadd_ps`.
- Transcendentals: `sqrtf` is exact via `float32(math.Sqrt(float64(x)))`.
  AEC3 only reaches libm through `exp2f` (from `std::pow(2.0f, x)`),
  `__exp10f` (from `powf(10.f, x)`) and `log10f` (metrics); use
  `Exp2f`/`Exp10f`/`Log10f` in `internal/aec3/cmath.go`. The oracle binary
  interposes those same Go functions for the libm symbols, because host libm float
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

- Use Go 1.27.4 (`$HOME/sdk/go1.27.4/bin/go`, or `go1.27.4`). The `Makefile`
  wraps every command below. CI (`.github/workflows/ci.yml`): `lint`;
  `test` (default, simd, purego on linux/amd64 and linux/arm64); `race`
  (`go test -race -short ./...` without and with simd); `oracle-src` (the
  source oracle in the AVX2, SSE2, NEON and scalar lanes, with and without
  simd; the avx2-simd job also runs the x86 kernel oracle, and the avx2-simd
  and neon-simd jobs log the glibc stock-libm divergence); `macos` (one
  macos-15 job: the tests in all three builds, the NEON and scalar source
  oracle and the Apple libm stock gate). Runs on main, nightly and manual
  runs are never cancelled; a new push to a pull request cancels its run.
- `GOEXPERIMENT=simd go test ./...` — default lane with archsimd kernels
  (`make test-simd`).
- `go test ./...` — default lane, pure-Go lane references (no experiment)
  (`make test`).
- `go test -tags purego ./...` — scalar lane (`make test-purego`).
- `GOARCH=amd64 GOEXPERIMENT=simd go test ./...` — amd64 lane (Rosetta;
  AVX2 available) (`make test-amd64`).
- `go test -race -short ./...`, with and without `GOEXPERIMENT=simd`
  (`make test-race`). `-short` scales the slow robustness tests down so the
  race run stays within the default timeout; the full run is unchanged.
- `make lint` — gofmt, `go vet` in every build configuration, and the
  generated source-oracle files are current.
- `GOEXPERIMENT=simd go test -tags webrtc_oracle ./internal/webrtcoracle/` —
  parity against the prebuilt libwebrtc (darwin/arm64 only; needs
  `tmp_check/libwebrtc`) (`make oracle`).
- `make oracle-stock` — `TestEc3StockLibmDivergence` against the prebuilt
  libwebrtc linked against Apple libm, without the libm interposer (tag
  `webrtc_oracle_stock`): every output, linear output and metrics frame must
  be bit-exact. `make oracle-src-stock` does the same against the source
  build on the host libm; it is gated on darwin (Apple libm) and only logs
  the divergence elsewhere (glibc's `log10f` moves the ERL metric by about
  1e-6 dB; the audio stays bit-exact).
- `scripts/fetch-webrtc.sh` (`make fetch-webrtc`) — sparse checkout of the
  pinned WebRTC sources and the abseil-cpp they include into
  `tmp_check/webrtc-m141`; idempotent.
- `CC=clang CXX=clang++ GOEXPERIMENT=simd go test -tags webrtc_src_oracle
  ./internal/webrtcoracle/` (`make oracle-src`) — end-to-end (BlockProcessor,
  EchoCanceller3, public API, the random-config differential test, the
  `FuzzEc3Differential` seeds, NaN-input parity) and per-kernel (matched
  filter, adaptive filter)
  parity against the C++ reference compiled from `tmp_check/webrtc-m141` on
  any OS/arch, in the machine's lane: NEON on arm64, AVX2 or SSE2 on amd64.
  The C++ lane follows `aec3.DetectOptimization()`; `GODEBUG=cpu.avx2=off`
  moves both to SSE2, `-tags purego` builds the scalar reference
  (`make oracle-src-scalar`). `GOAEC3_EXPECT_LANE=none|sse2|avx2|neon` fails
  the run unless both sides run that lane. Needs clang. The `src_gen_*` files
  are generated by `scripts/gen-src-oracle.sh` (copies of the `webrtc_oracle`
  ec3 tests and shim with only the build constraint changed, and one
  translation unit per WebRTC source); regenerate after editing the
  originals.
- x86 lanes on Apple silicon: `make oracle-src-rosetta` runs the source
  build for x86_64 under Rosetta (`CGO_ENABLED=1 GOARCH=amd64 CC="clang -arch
  x86_64" CXX="clang++ -arch x86_64"`): the SSE2 and scalar lanes;
  `make oracle-src-rosetta-avx2` runs the AVX2 lane with
  `GOAEC3_FORCE_LANE=avx2` (Rosetta hides AVX and BMI2 from CPUID) and
  `GOAEC3_FORCE_FMA=1` (Rosetta executes FMA but CPUID does not report it).
- `GOEXPERIMENT=simd go test -tags ooura_sse2_oracle ./internal/webrtcoracle/`
  (`make oracle-x86-kernels`) — the x86 Ooura FFT (scalar and SSE2),
  FftData::Spectrum and VectorMath (SSE2 and AVX2) kernels only, compiled
  from `tmp_check/webrtc-m141`: natively on linux/amd64 (`CC=clang
  CXX=clang++`), under Rosetta on darwin (`CGO_ENABLED=1 GOARCH=amd64
  CC="clang -arch x86_64" CXX="clang++ -arch x86_64"`; `GOAEC3_FORCE_FMA=1`
  forces the FMA kernel tests).

## Layout

- `internal/aec3` — the AEC3 core (one package, like the C++ directory).
- `internal/bands` — the band splitting around AEC3: the 2-band QMF
  (32 kHz), the 3-band filter bank (48 kHz) and AudioBuffer's FloatS16
  conversions.
- `internal/ooura` — 128-point Ooura real FFT (scalar + NEON/SSE2 lanes).
- `internal/webrtcoracle` — cgo oracle harness (tags `webrtc_oracle`,
  `webrtc_src_oracle`, `ooura_sse2_oracle`; `webrtc_oracle_stock` with
  either oracle tag links the host libm instead of the interposer).
- root package `goaec3` — public API.
