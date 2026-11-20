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
- Where a transcription of the intrinsics is slow as Go code, the fallback is
  a second pure-Go implementation written for the Go compiler (scalar locals,
  proven bounds, fused passes; `internal/ooura/fft_fast.go`), tested
  bit-exactly against the transcription, which stays the spec.
- archsimd semantics: `a.Mul(b).Add(c)` compiles to separate multiply and add
  (unfused, like `vmlaq_f32`/`_mm_mul_ps`+`_mm_add_ps`); `a.MulAdd(b, c)` is a
  real FMA (like `vfmaq_f32`/`_mm256_fmadd_ps`). Use each exactly where the C++
  does.
- No hand-written assembly. Kernels are archsimd; a kernel archsimd cannot
  express stays pure Go. Gate x86 kernels with `archsimd.X86.AVX2()` etc.
- Every vector instruction runs behind a test of its CPU feature: a branch
  on a package-level variable set from `archsimd.X86` (`kernHasAVX2`,
  `dlyUseFma`, ...), in the function itself or in every caller. A lane or
  group-size argument does not count; the kernel it selects tests the
  variable again. `TestSimdInstructionsGated` proves this from the compiler's
  listing of the amd64 build, tests included (a test that calls a kernel
  directly skips, or leaves the kernel out of its table, behind the same
  variable), and the `avx-only` CI job runs the tests on an emulated CPU
  with AVX and no AVX2.
- 256-bit code needs AVX2, also where its instructions are AVX: without AVX2
  Go 1.27 saves only the low halves of the Y registers at an asynchronous
  preemption (golang/go#81209).
- The 128-bit kernels run with AVX alone, and some 128-bit archsimd
  operations compile to AVX2 instructions: `BroadcastFloat32x4`,
  `BroadcastInt32x4`, `BroadcastUint32x4` (VBROADCASTSS from a register,
  VPBROADCASTD) and `Float32x4.Abs`/`Neg`, which broadcast their mask. Use
  `kernSplat4`/`kernSplatI4` (`bsSplat4`/`bsSplatU4` in `internal/bands`) and
  a mask made with them. A VEX mnemonic the kernels did not use before fails
  the test until it is classified in `simdVexOps`.
- End every 256-bit archsimd section with `archsimd.ClearAVXUpperBits()`
  (VZEROUPPER). Go's scalar float code is legacy-SSE encoded, so a dirty upper
  YMM state makes every later scalar float op pay the AVX→SSE transition
  penalty: on Intel this made the AVX2 lane 16× slower per frame.
  `TestNoLegacySSEWithDirtyUpperState` follows the state along every path of
  the amd64 listing and fails on a legacy-SSE instruction (a scalar float
  operation, a zero made with `MOVUPS X15`, a call out of the kernels)
  between a 256-bit instruction and the VZEROUPPER.
- A float32 live across `ClearAVXUpperBits` is spilled with a 4-byte store
  and restored with a 16-byte load, which the CPU cannot store-forward (this
  made Zen 3 1.6× slower). 256-bit kernels return results through the
  caller's memory, or keep their 256-bit part in a `//go:noinline` helper.
- Scalar float code inside a 256-bit section pays the same transition
  penalty as after it: a kernel takes the scalars it needs from vector
  lanes, or computes them before its first 256-bit operation or after
  `ClearAVXUpperBits` (`internal/ooura/fft_pair_simd_amd64.go`). Moves of
  single floats count: the AVX2 three-band synthesis filters into a stack
  array and interleaves it after `ClearAVXUpperBits`, which made it four
  times faster on Zen 3.
- amd64 folds only a constant displacement into an archsimd load or store;
  an offset that depends on a loop counter costs one or two LEAs per access.
  Loops with a fixed trip count are written as a closure that captures a
  vector and is called once per iteration with constant arguments: the
  compiler inlines closures that capture SIMD values (sixteen times the
  usual budget), and every access becomes base plus displacement
  (`cftfsubTailSimd`, `afirAdaptTerms`). Such closures nest (a closure that
  calls another four times still inlines); a function that takes the closure
  as an argument does not inline it. Check with `-gcflags=-S` that no
  `funcN` call remains.
- The slice helpers of `kern_bins*.go` are called on a block (64 samples) or
  a spectrum (65 bins). A loop over slices pays a bounds check and an
  address computation per slice and step, several times the vector
  operation, so for those lengths a helper stores the 65th element and runs
  its 64-element form (`kern_bins_simd64.go`; 256-bit forms in
  `kern_bins_simd_amd64.go`), written with constant-offset closures.
- A loop over slices of any length gets the same addressing by advancing
  the slices: the loop tests `len(s) >= 64` for each slice, takes
  `(*[64]float32)(s)` and ends with `s = s[64:]`, so the block pointer is
  the slice pointer and every access is pointer plus constant. An array
  pointer taken at `s[i : i+16]` with a loop counter `i` costs two LEAs per
  access instead (the Sum kernels of the matched filter,
  `matched_filter_simd_amd64.go`).
- The term kernels of the adaptive filter (`afirApplyTerms`,
  `afirAdaptTerms`) pass over their (X, H) pairs once per block of bins, so
  the per-pair bookkeeping runs four to eight times per call and the loop is
  bound by instruction count: the pairs are listed once as two pointer
  arrays on the stack (16 entries for short filters, zeroed per call) and a
  pass loads two pointers per pair. Stepping the ring and the filter inside
  the passes cost 26% more per ApplyFilter on Zen 3.
- A loop that indexes a slice field of its receiver and stores through it
  reloads the slice and checks the index on every access, because the store
  may alias the receiver: take the element array through a local pointer
  first (`SubbandErleEstimator`).
- Sums that the C++ accumulates in element order keep that order; sums that
  are independent of each other run side by side in one loop
  (`FullBandErleEstimator.Update`), which divides the wait for the chains.
- The FFT entry points of aec3 are `ooura.FftSplit`/`InverseFftSplit`: they
  move between a time-domain frame and the real/imaginary arrays of
  `FftData` directly (the last forward pass and the first inverse pass work
  on deinterleaved vectors anyway), and write every result where the next
  aligned vector load reads it, so no load waits on two smaller stores.
  `FftSplitPair`/`InverseFftSplitPair` run two transforms in one 256-bit pass
  on AVX2 (each 128-bit half is one transform in the SSE2 order) and are two
  single transforms elsewhere; aec3 pairs the transforms that come in twos
  (`aec3FftPair`). Their work areas live with the caller (`ooura.Scratch`,
  `PairScratch`).
- On arm64 the vector operations decide. A Neoverse-N2 has two
  floating-point pipes and runs a kernel at the pace of its floating-point
  and vector operations, shuffles, scalar float operations and (in part)
  vector stores included; loads and integer work ride along. Count them in
  `-gcflags=-S` output: a kernel with fewer of them is faster in proportion.
  - The split transforms of arm64 (`fft_soa_simd_arm64.go`) keep the real
    and the imaginary parts in separate vectors: a rotation needs no lane
    swap, the bit reversal is the 4x4 transposition between the first two
    passes, and the transform has a quarter fewer vector operations than the
    passes of `fft_simd.go`, which remain the packed transforms and the
    amd64 SSE2 lane.
  - Scalar float code costs per operation what a vector of four costs. The
    decimation by 4 (`decimator_simd_arm64.go`) runs its four cascaded
    biquad sections as the lanes of one vector, section i two samples behind
    section i-1: ten vector operations per sample instead of 36 scalar ones.
    The two samples keep the path through a lane's input (a shuffle, a
    product, four additions) off the critical cycle, which is the path from
    an output of a section to its next one. The samples the lanes do not
    cover run through the scalar section, and a block with a value that is
    not finite runs the scalar sections (the operand order of an addition
    only shows in the payload of a NaN).
  - A closure that captures no vector is not inlined above the usual budget:
    `_ = zero` on a captured vector gives it the budget of a kernel closure.
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

## Pure-Go lane references (no GOEXPERIMENT)

These rules come from the matched filter, which is about half of a pure-Go
frame (`matched_filter_kernels.go`, `matched_filter_scalar.go`).

- A fused float32 multiply-add is `fl64(a*b + c)` narrowed to float32 unless
  the float64 sum is a float32 rounding boundary (`fmafTie` in `fmaf.go`).
  A loop tests every sum, leaves on a hit and hands the work to a
  `//go:noinline` function that calls `fmaf` per element and restarts from
  what the loop has stored, so the loop keeps no value alive for it.
  `math.FMA` on the widened operands returns the same float64 sum (the
  product is exact) and removes neither the test nor a conversion.
- The cost of that emulation on amd64 is its instruction count (about twenty
  per fused update and accumulation, against eight for the unfused lanes):
  a form with fewer conversions or loads is faster in proportion, a form
  that adds a store and a load to convert in place is slower.
- Loops that read an operand many times read a float64 copy that is widened
  once (`dlyWide`: the downsampled render buffer, once per block). The copy
  is a pointer-free array that is the last field of the block processor and
  reaches the matched filter as an argument (`GetDelayWide`), so it adds no
  heap object and no pointer word and sits behind the canceller's last
  pointer.
- amd64 has fifteen float registers, and the compiler schedules a value
  without a use in its own basic block (the addition of an accumulator)
  after every value that has one (the products). A block with eight chains
  spills one of them; split the body into blocks (a test between half
  groups, the tie test of each fused sum, a bounds check).
- A chain of additions costs the latency of an addition per step however
  idle the CPU is. Independent chains run side by side: the scalar lane runs
  up to three filters in step (`dlyCoreScalar`), the x*x sums of the unfused
  lanes run eight chains over a table of squares.
- Work that cannot change state is not done: Update returns early when the
  render buffer is quiet, skips the core of a filter whose windows are quiet
  (`filterIsQuiet`), and the cores that have their x*x sums first skip the
  h*x sums of a filter that no sample of the block updates
  (`dlyLaneFilter.idle`), and under a zero capture block the cores of
  filters that are zero (`filterStaysZero`); the arguments are in
  `matched_filter.go`. Blocks
  whose windows do not wrap run without the list of stretches
  (`dlyCoreDirect`).
- Where the unfused lanes are bound by the instruction count (two multiplies
  and two additions per coefficient), the reference is at its floor; only
  the archsimd kernels go below it.
- Timings on a shared Mac and under Rosetta do not predict x86: decide from
  the kernel benchmarks of `bench.yml`, which run base and head on the same
  runner.

The rest of a pure-Go frame (adaptive filter, FFT, filter bank, bin loops):

- The term kernel of ApplyFilter and the adaptation are bound by their loads
  and float operations, not by their instruction count. Fewer instructions
  per bin gain little (eight bins per step: 1% and 5% on Neoverse-N2), and a
  form that loads an operand twice loses: two passes over the pairs with
  half the live values each, every load folded into a multiplication, are 17%
  (Xeon) to 33% (EPYC) slower than one pass that spills
  (`afirApplyTermsQuadGo`).
- What gains is a load and a store less per value. Passes that each load and
  store a whole frame run as one pass where their operands meet: the split
  transforms run cftfsubTail, rftfsub and the copy to bins as one pass and
  the copy from bins, rftbsub and cft1st as one (`tailSplitFast`,
  `splitCft1stFastLane`; the butterflies at c and 16-c hold each other's
  rftfsub partners), 15-19% per transform on every runner. Off amd64 the
  filter bank takes two output samples per step, which loads a coefficient
  once and the history in pairs (9-11% on Neoverse-N2).
- A float comparison that selects a float compiles to a branch on arm64 and
  amd64 (`stdMax`, `sseMax`), and on data it mispredicts. Where the operands
  are known to be +0, positive or +Inf, the maximum is the larger bit pattern
  as an unsigned integer, a conditional move, and a pattern above that of
  +Inf marks the NaN, negative or -0 operand that takes the comparison
  (`afirFreqRespBlockSse2Fast`, `kernMaxAbsBits`). A maximum with a value
  known to be +0 is not computed (`afirFreqRespFirstGo`).
- An addition of +0 that only normalizes -0 is dropped where every use of
  the sum is an addition to a sum that is never -0 (the filter outputs of
  the three-band filter bank).
- A helper called several times in a loop has to inline: a helper that
  returns two results of four products each does not, two calls of one that
  returns one do, and the compiler shares their loads (`bsTapAt`).

## Memory safety: no unsafe

- No non-test file imports `unsafe` (so there is no `unsafe.Pointer`,
  `unsafe.Add`, `unsafe.Slice` or `//go:linkname`). `TestNoUnsafe` in the root
  package scans every Go file of the module, whatever its build tags, and
  fails on the import. The only exception is `internal/webrtcoracle`, the cgo
  test oracle, which no shipped package imports.
- Unchecked access is obtained by proof, not by pointer arithmetic: convert a
  slice to an array pointer over a window with constant bounds
  (`(*[32]float32)(a[64:96])`), then index or re-slice it with a constant, a
  masked value, or a loop counter with constant limits (`w[j : j+4]`,
  `w[8*m : 8*m+8]`). Write the window as `x : x+c`; the compiler does not
  prove `x+c1 : x+c2`. archsimd loads and stores take the resulting
  `*[N]float32` (`LoadFloat32x4Array`, `StoreArray`).
- Check with `go build -gcflags=-d=ssa/check_bce ./internal/...` (it lists the
  checks left in each function; a helper that inlines everywhere still lists
  the checks of its own never-called body) or look for `panicBounds` calls in
  `-gcflags=-S` output. `TestNoBoundsChecks` in `internal/ooura` does the
  latter for every FFT pass, on arm64 and amd64, with and without the
  experiment.
- Reinterpreting bits goes through `math.Float32bits`/`Float32frombits` or
  archsimd's `ToBits`/`BitsToFloat32`, never through a pointer cast.

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
`testing.AllocsPerRun(...) == 0` test after warm-up. The one exception is the
render transfer queue: it starts small and grows to the deepest render burst
so far (up to `RenderTransferQueueSizeFrames`), so a render burst deeper than
any before grows the queue once; allocation tests warm up past the deepest
burst they measure.

## Garbage collector footprint

A canceller is long-lived state the garbage collector marks on every cycle.
What the mark costs: the heap objects (about 25 ns each), the pointer words
it reads (every pointer word of a live object, nil or not, and about 15 ns
more for one that is not nil, also when it points into an object that is
already marked), and the length of an object up to its last pointer word
(about 5 ns per KB). A pointer-free object is marked without being read,
whatever its size. So a canceller is one object whose few pointer words are
its first, and one pointer-free object for its arrays (the arena).

- A component that needs a pointer (a slice whose size depends on the
  config, the state of more than `inlineChannels` channels, an optional
  component) is two types (`internal/aec3/arena.go`): `xState`, a
  pointer-free struct that the state of its parent holds by value, and `X`,
  a view: `*xState` next to the slices and the views of its children. The
  methods are those of the view and the fields of the state are promoted
  into it, so a method body reads as if there were one struct.
- `Ec3Core` is what a canceller keeps: the pointer words (the arena, and
  those that are nil in the usual canceller: the grown render transfer
  queue, the extras) and the state of every component. Every entry point
  binds the views on its stack
  (`Ec3Core.Bind`, `BindRender`) and calls their methods. A view must not
  escape: it is not stored in an interface or a closure (the allocation
  tests show it).
- `init` allocates the slices of a view as if it stood alone. `Ec3Core.adopt`
  then walks the views twice with a binder (`bind`, one method per view
  type that lists its slices and children): it records the length of every
  slice in a `span` of the state and moves the slices into the arena, which
  has one array per element type (`[]float32`, `[]FftData`, `[][65]float32`,
  `[]int32`). A slice a view registers owns its memory; a slice whose length
  changes while processing keeps the length in the state (`Subtractor`).
- The arena (`aec3.Arena`) is one object behind an interface, so the
  canceller holds one pointer whatever the number of arrays. The public
  package allocates it as a struct of arrays (an arena class, `arena.go`):
  the classes are sized for the default configs with up to two channels a
  side (`TestArenaClassOfDefaults`), and a canceller takes the smallest that
  holds what it needs; one that fits no class gets an arena of slices
  (`NewSliceArena`), as the views of `NewEchoCanceller3` do. The arena also
  holds the band-splitting filters and the full-band copies of the public
  package, and the first frames of the render transfer queue. A canceller
  with several render channels sets its block processor up again, at
  another size, when stereo content starts or ends: the new arrays go into
  the arena when they fit and into a second arena, which lives as long as
  that block processor, when they do not (stereo content in a canceller
  that started without).
- A view made by `NewX` stands alone (`bind(st, bindAlone)`): tests and the
  oracle harness use the components as before.
- A method of a view reaches a field of the state through the view's
  pointer to it, which is one load more than a field of its own, and after
  a store the compiler loads that pointer again. The small views (the
  filters, the rings, the analyzers and estimators) therefore have value
  receivers, which keep the pointer in a register; only the methods that
  set a field of the view itself (`init`, `bind`, a slice that is replaced)
  have pointer receivers. A view keeps a copy of the fields of its state
  that never change and that its accessors read (the dimensions of a ring).
  `TestEc3ViewPerCall` runs a view bound per call against a view bound once,
  bit for bit: whatever changes between calls must be in the state.
- Per-channel state of up to `inlineChannels` capture channels lives in
  arrays of the state; the component derives the slices on use
  (`channelState`). What more channels need, and the optional components,
  are heap objects behind one pointer each (`more`), which the canceller
  keeps in `bpExtra`: a canceller with up to `inlineChannels` channels and
  the default components has none. `TestEc3ChannelStateStorage` runs both
  forms against each other.
- Multi-dimensional buffers are one pointer-free array with views computed
  from the indices (`Frame.View`, `Block.View`, the rings' `At`, the matched
  filter's `filter(n)`), not slices of slices.
- A component copies the config (as the C++ does) rather than pointing at
  it. An optional implementation is a tagged value, not an interface
  (`stTransparentState`).
- The render side runs concurrently with the capture side: it reads only
  the pointers that are set once (`arena`, `hpMore`) and the queue. The
  arena of the block processor changes on the capture side.
- The tests: `TestPointerWords` (the pointer words of the canceller type and
  where the last one ends: the state has none), `TestPointerSlots` (the
  pointers of a canceller), `TestNewFootprint` (its objects: two), `TestArenaClasses`,
  `TestGCMarkPerCanceller` (the mark time a canceller adds to a cycle) and
  the `Test...PointerFree` tests of `internal/aec3`. `BenchmarkGCMark -cpu 1`
  times a cycle with 1000 cancellers; `net-mark-ns/op` is the mark phase
  less that of a cycle without them.

## Build and test

- Use Go 1.27.4 (`$HOME/sdk/go1.27.4/bin/go`, or `go1.27.4`). The `Makefile`
  wraps every command below. CI (`.github/workflows/ci.yml`): `lint`;
  `test` (default, simd, purego on linux/amd64 and linux/arm64); `race`
  (`go test -race -short ./...` without and with simd); `avx-only` (the
  short simd tests under `qemu-x86_64 -cpu SandyBridge`); `oracle-src` (the
  source oracle in the AVX2, SSE2, NEON and scalar lanes, with and without
  simd; the avx2-simd job also runs the x86 kernel oracle, and the avx2-simd
  and neon-simd jobs log the glibc stock-libm divergence). Runs on main,
  nightly and manual runs are never cancelled; a new push to a pull request
  cancels its run. `.github/workflows/macos.yml` runs separately (one
  macos-15 job: the tests in all three builds, the NEON and scalar source
  oracle and the Apple libm stock gate), so a macOS job waiting for a runner
  never holds up ci; a newer macOS run cancels a waiting one.
  `.github/workflows/bench.yml` times Go against the source-built libwebrtc
  per lane on real x86 and arm64 runners (push to `perf/**` or manual).
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
