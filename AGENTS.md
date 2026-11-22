# Agent Instructions

goaec3 is a pure-Go port of WebRTC's AEC3 echo canceller
(`modules/audio_processing/aec3`), bit-exact with libwebrtc m141
(`branch-heads/7390`, commit `d2eaa5570fc9959f8dbde32912a16366b8ee75f4`).

Write code comments and docs in the present tense, in the style of the Go
standard library: what the code does and why, never how it got there, and no
benchmark numbers.

## Parity

- Every ported module has a differential test against the C++ oracle
  (`internal/webrtcoracle`) before another module builds on it. Kernels are
  bit-exact (`math.Float32bits` equality), in output and in state.
- Never weaken a gate, a tolerance or a test bound.
- Work the C++ does is skipped only where output and state are provably
  identical; the argument stands at the skip and a test covers it.
- The reference sources live in `tmp_check/webrtc-m141` (not committed;
  `make fetch-webrtc`). `tmp_check/libwebrtc` links the prebuilt m141
  `libwebrtc.a` (darwin/arm64, NEON, `-ffp-contract=off`). The oracle
  compiles what the prebuilt library lacks (the scalar path) from the
  sources under renamed symbols, with `#pragma clang fp contract(off)`.

## Lanes

- `purego` tag: the scalar C++ path (`Aec3Optimization::kNone`, scalar Ooura
  FFT) on every architecture.
- Default build: the SIMD path libwebrtc selects, NEON on arm64, SSE2 or AVX2
  on amd64. The SIMD variants reorder reductions; each lane follows its own
  operation order.
- `DetectOptimization` is one routine for builds with and without
  `GOEXPERIMENT=simd` and uses libwebrtc's kAVX2 condition (AVX, OSXSAVE/YMM,
  AVX2, BMI2), so both builds pick the same lane.
- The library never reads the environment. Tests force a lane with
  `aec3.SetForcedLaneForTesting`; the source oracle's TestMain applies
  `GOAEC3_FORCE_LANE` through it.
- A C++ class that takes an `Aec3Optimization` keeps the argument in Go, so
  one binary tests every lane.

## Kernels

Toolchain: Go 1.27.4. `simd/archsimd` exists only with `GOEXPERIMENT=simd`.

- No assembly. A kernel archsimd cannot express stays pure Go.
- Files: `foo_simd.go` (`goexperiment.simd && (arm64 || amd64) && !purego`),
  split per architecture where the lane order differs, and `foo_nosimd.go`
  (the negation), which routes to the pure-Go lane reference.
- Every lane has a pure-Go reference (`fooNeonRef`, `fooSse2Ref`): the spec,
  the fallback without the experiment, and what the archsimd kernel is tested
  against bit for bit. Where the transcription is slow, a second pure-Go
  form written for the compiler is the fallback, tested bit for bit against
  the transcription (`internal/ooura/fft_fast.go`).
- `a.Mul(b).Add(c)` is unfused (`vmlaq_f32`, `_mm_mul_ps` + `_mm_add_ps`);
  `a.MulAdd(b, c)` is an FMA (`vfmaq_f32`, `_mm256_fmadd_ps`). Use each where
  the C++ does.

### amd64

- Every vector instruction runs behind a test of its CPU feature: a branch
  on a package variable set from `archsimd.X86` (`kernHasAVX2`, `dlyUseFma`),
  in the function or in every caller. A lane argument does not count.
  `TestSimdInstructionsGated` proves it from the compiler's listing, tests
  included: a test that calls a kernel directly skips behind the same
  variable. The `avx-only` CI job runs on an emulated CPU with AVX and no
  AVX2.
- 256-bit code needs AVX2 even where its instructions are AVX: without AVX2,
  Go 1.27 saves only the low halves of the Y registers at an asynchronous
  preemption (golang/go#81209).
- 128-bit kernels run with AVX alone, but `BroadcastFloat32x4`,
  `BroadcastInt32x4`, `BroadcastUint32x4`, `Float32x4.Abs` and `Neg` compile
  to AVX2 instructions. Use `kernSplat4`/`kernSplatI4` (`bsSplat4`/`bsSplatU4`
  in `internal/bands`) and masks made with them. A VEX mnemonic new to the
  kernels fails the test until it is classified in `simdVexOps`.
- End every 256-bit section with `archsimd.ClearAVXUpperBits()`. Go's scalar
  float code is legacy-SSE encoded and pays a transition penalty per
  operation while the upper state is dirty; `TestNoLegacySSEWithDirtyUpperState`
  fails on a legacy-SSE instruction between a 256-bit instruction and the
  VZEROUPPER. Scalar float code, single-float moves included, runs before the
  first 256-bit operation or after the clear; a kernel takes the scalars it
  needs from vector lanes.
- A float32 live across `ClearAVXUpperBits` is spilled with a 4-byte store
  and reloaded with a 16-byte load, which cannot store-forward. A 256-bit
  kernel returns results through the caller's memory or keeps its 256-bit
  part in a `//go:noinline` helper.
- Only a constant displacement folds into an archsimd load or store. A loop
  with a fixed trip count is a closure that captures a vector, called once
  per iteration with constant arguments; the compiler inlines closures that
  capture SIMD values, nested ones too, but not one passed as an argument. A
  closure that captures no vector gets the budget with `_ = zero` on one.
  Check with `-gcflags=-S` that no `funcN` call remains.
- A loop over slices of any length advances the slices: test
  `len(s) >= 64`, take `(*[64]float32)(s)`, end with `s = s[64:]`. An array
  pointer taken at `s[i:i+16]` costs two LEAs per access.
- The slice helpers of `kern_bins*.go` take a block (64) or a spectrum
  (65): they store the 65th element and run the 64-element form.
- The term kernels of the adaptive filter list their (X, H) pairs once, as
  two pointer arrays on the stack.

### arm64

- The count of floating-point and vector operations decides (shuffles,
  scalar float operations and vector stores included; loads and integer work
  do not). Count them in `-gcflags=-S` output.
- A scalar float operation costs what a vector of four costs: the decimator
  runs its four biquad sections as the lanes of one vector. A block with a
  non-finite value runs the scalar code (operand order shows in a NaN
  payload).
- A call of an inlined helper leaves a NOP unless its line has an operation
  of its own (`internal/ooura/fft_soa_simd.go`).

### Both

- The FFT entry points are `ooura.FftSplit`/`InverseFftSplit` and the `Pair`
  forms, which move between a frame and the arrays of `FftData`; the work
  areas live with the caller. `ooura.Fft`/`InverseFft` are the lane's
  transcription, for tests.
- A loop that indexes a slice field of its receiver and stores through it
  reloads the slice on every access: take the element array through a local
  pointer first.
- A sum the C++ accumulates in element order keeps that order; independent
  sums run side by side in one loop.

## Pure-Go lane references

- A fused float32 multiply-add is `float32(float64(a)*float64(b) + float64(c))`
  unless the float64 sum is a float32 rounding boundary (`fmafTie`). A loop
  tests every sum, leaves on a hit and hands the rest to a `//go:noinline`
  function that calls `fmaf` per element. `math.FMA` on the widened
  operands removes neither the test nor a conversion. The emulation costs
  its instruction count: fewer conversions or loads win, a store and a load
  to convert in place lose.
- An operand read many times is read from a float64 copy widened once
  (`dlyWide`), a pointer-free array behind the canceller's last pointer.
- amd64 has fifteen float registers: a basic block with eight live chains
  spills, so split the body. A chain of additions costs one latency per
  step: run independent chains side by side.
- Skip work that cannot change state (`filterIsQuiet`, `dlyLaneFilter.idle`,
  `filterStaysZero`); the arguments are in `matched_filter.go`.
- The unfused lanes are at their instruction floor (two multiplies and two
  additions per coefficient); only the archsimd kernels go below it.
- The adaptive filter kernels are bound by loads and float operations: one
  pass that spills beats two passes that load an operand twice. Fuse passes
  that each load and store a whole frame (`tailSplitFast`,
  `splitCft1stFastLane`).
- A float comparison that selects a float is a branch. Where the operands
  are +0, positive or +Inf, take the maximum of the bit patterns; a pattern
  above +Inf's marks the operand that needs the comparison
  (`kernMaxAbsBits`). A maximum with a value known to be +0 is not computed.
- An addition of +0 that only normalizes -0 is dropped where every use adds
  the sum to one that is never -0.
- A helper called several times in a loop must inline; one that returns two
  results of four products each does not (`bsTapAt`).
- Local and Rosetta timings do not predict x86. Decide from `bench.yml`,
  which runs base and head on one runner.

## No unsafe

- No non-test file imports `unsafe` (`TestNoUnsafe`); the exception is
  `internal/webrtcoracle`.
- Bounds checks go by proof: convert a slice to an array pointer over a
  constant window (`(*[32]float32)(a[64:96])`), then index it with a
  constant, a masked value or a counter with constant limits. Write windows
  as `x : x+c`; the compiler does not prove `x+c1 : x+c2`.
- Check with `go build -gcflags=-d=ssa/check_bce ./internal/...`;
  `TestNoBoundsChecks` covers the FFT passes. Reinterpret bits with
  `math.Float32bits`/`Float32frombits` or archsimd's `ToBits`/`BitsToFloat32`.

## Floats

- Signal storage and arithmetic are `float32`. Use `float64` only where the
  C++ uses `double`.
- Go fuses `x*y + z` on arm64, across statements too; libwebrtc is built
  with contraction off. Wrap every product that feeds an add or subtract in
  `float32(...)`, products stored in locals included.
- `sqrtf` is `float32(math.Sqrt(float64(x)))`. The other libm calls AEC3
  makes go through `Exp2f`, `Exp10f` and `Log10f` (`internal/aec3/cmath.go`),
  which the oracle interposes for the libm symbols: host libms are not
  correctly rounded and differ. A new libm call needs a helper there and an
  interposer in `internal/webrtcoracle/libm.go`.

## Naming

- One Go file per C++ file. A C++ class is a Go type of the same name with
  constructor `NewX` and exported methods of the C++ names. `kFoo` drops the
  `k`. `std::array<float, N>` is `[N]float32`; `size_t` and `int` are `int`.
- `EchoCanceller3Config` fields are the C++ names in CamelCase.
- ApmDataDumper, metrics reporting and logging are not ported. Field trials
  are fixed at their defaults.

## Allocation

Steady-state processing does not allocate; a new hot path gets a
`testing.AllocsPerRun(...) == 0` test. The render transfer queue grows once
per render burst deeper than any before, so allocation tests warm up past
their deepest burst.

## Garbage collector

The mark cost of a canceller is its heap objects, its pointer words (nil or
not) and the length of each object up to its last pointer word. A
pointer-free object is not read. A canceller is one object whose few pointer
words come first, and one pointer-free arena for its arrays.

- A component that needs a pointer is two types (`internal/aec3/arena.go`):
  `xState`, pointer-free and held by value in its parent's state, and the
  view `X`, a `*xState` next to slices and child views. Methods belong to the
  view; the fields of the state are promoted into it.
- `Ec3Core` holds the pointer words and every state. Entry points bind the
  views on their stack (`Bind`, `BindRender`). A view does not escape: no
  interface, no closure.
- `init` allocates a view's slices as if it stood alone; `Ec3Core.adopt`
  moves them into the arena through each view's `bind`. A slice whose length
  changes while processing keeps the length in the state. A view made by
  `NewX` stands alone, for tests and the oracle.
- The public package allocates the arena as a struct of arrays in size
  classes (`arena.go`, `TestArenaClassOfDefaults`); a canceller that fits no
  class gets `NewSliceArena`. A block processor set up again when stereo
  content starts or ends goes into the arena if it fits and into a second
  arena otherwise.
- Small views have value receivers, which keep the state pointer in a
  register; only methods that set a field of the view have pointer
  receivers. A view may copy fields of its state that never change.
  Whatever changes between calls is in the state (`TestEc3ViewPerCall`).
- State of up to `inlineChannels` capture channels lives in arrays of the
  state; more channels and optional components are heap objects behind one
  pointer each (`TestEc3ChannelStateStorage`).
- Multi-dimensional buffers are one array with computed views, not slices
  of slices. A component copies the config. An optional implementation is a
  tagged value, not an interface.
- The render side runs concurrently with the capture side and reads only
  pointers that are set once and the queue.
- The bounds of `TestPointerWords`, `TestPointerSlots`, `TestNewFootprint`
  and `TestGCMarkPerCanceller` only go down. `BenchmarkGCMark -cpu 1` times
  a cycle with 1000 cancellers.

## Build and test

The `Makefile` wraps every command; CI runs them (`.github/workflows`).

- `make test`, `test-simd`, `test-purego`, `test-amd64`, `test-race`: the
  unit tests per build. `make lint`: gofmt, `go vet` per build, generated
  oracle files current (`scripts/gen-src-oracle.sh`).
- `make oracle`: parity against the prebuilt libwebrtc (darwin/arm64).
- `make oracle-src`, `oracle-src-scalar`: parity against the C++ compiled
  from the sources, in the machine's lane; needs clang.
  `GODEBUG=cpu.avx2=off` moves both sides to SSE2; `GOAEC3_EXPECT_LANE`
  fails the run unless both sides run the named lane.
- `make oracle-src-rosetta`, `oracle-src-rosetta-avx2`: the x86 lanes on
  Apple silicon. Rosetta hides AVX, BMI2 and FMA from CPUID, hence
  `GOAEC3_FORCE_LANE=avx2` and `GOAEC3_FORCE_FMA=1`.
- `make oracle-x86-kernels`: the x86 FFT, spectrum and vector-math kernels
  against their C++ sources.
- `make oracle-stock`, `oracle-src-stock`: against the C++ on the host libm.
  Gated on darwin; glibc's `log10f` moves the ERL metric and is only logged.
- `bench.yml` times Go against the source-built libwebrtc per lane, base and
  head on one runner (push to `perf/**`).

## Layout

`internal/aec3` is the AEC3 core, one package like the C++ directory;
`internal/bands` the band splitting and AudioBuffer's FloatS16 conversions;
`internal/ooura` the 128-point Ooura FFT; `internal/webrtcoracle` the cgo
oracle (tags `webrtc_oracle`, `webrtc_src_oracle`, `ooura_sse2_oracle`,
`webrtc_oracle_stock`); the root package `goaec3` the public API.
