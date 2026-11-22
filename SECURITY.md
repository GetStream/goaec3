# Security

## Supported versions

goaec3 is pre-v1. Fixes go to `main` and the latest tag only.

## Reporting a vulnerability

Report it privately: on GitHub, open the repository's **Security** tab and
choose **Report a vulnerability**. Please don't open a public issue. Include
a config and input that reproduce the problem if you can.

## What goaec3 defends against

- **Untrusted audio is fine.** No frame makes a call panic. A frame of the
  wrong shape returns an error and changes nothing. Every `float32` value is
  accepted; out-of-range values and ±Inf are clamped.
- **NaN is not an attack, but it is permanent.** As in libwebrtc, one NaN
  sample makes every later output NaN. Replace NaN before the call, or create
  a new canceller.
- **Untrusted configs are bounded.** `New` returns an error for every config
  it cannot run. What it accepts is limited to 16 channels a side
  (`MaxChannels`), filters of 250 blocks (`MaxFilterLengthBlocks`), 64 delay
  filters (`MaxDelayFilters`) and the ranges of `Config.Validate`. The largest
  canceller takes less than 64 MB and about a hundred times the default's
  work per frame (tens of milliseconds in pure Go). That is bounded, not
  real time: a service that takes configs from its users should set lower
  limits of its own.
- **Memory is bounded per canceller.** Nothing grows with the audio. The
  render queue holds at most 100 frames; further render frames are dropped.
- **No `unsafe`, no cgo, no assembly** in the library. One dependency,
  `golang.org/x/sys`, for CPU detection.

## What it does not defend against

- **Breaking the concurrency contract.** `AnalyzeRender` may run next to the
  capture-side calls; two concurrent calls on the same side are a data race.
  Nothing detects it. The output is garbage and the call may panic.
- **Timing.** Processing time depends on the audio and the config. goaec3 is
  not constant time and makes no attempt to be.
- **A zero `EchoCanceller`.** Create cancellers with `New`.

## How this is checked

Unit tests and a fuzzer drive the public API with random configs, call
orders and audio, in every build. CI runs the race detector and
`govulncheck`; actions are pinned to commits.
