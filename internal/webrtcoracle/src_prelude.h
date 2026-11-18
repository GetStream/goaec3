// Prelude of every translation unit of the webrtc_src_oracle build.
#ifndef GOAEC3_SRC_PRELUDE_H
#define GOAEC3_SRC_PRELUDE_H

// libwebrtc is built by clang with -ffp-contract=off. gcc ignores the pragma
// below and contracts by default, so the build insists on clang.
#if !defined(__clang__)
#error "the webrtc_src_oracle build needs clang (CC=clang CXX=clang++)"
#endif
#if !defined(NDEBUG)
#error "the webrtc_src_oracle build is a release (NDEBUG) build, like libwebrtc"
#endif

#pragma clang fp contract(off)
#pragma clang diagnostic ignored "-Wnullability-completeness"

#endif  // GOAEC3_SRC_PRELUDE_H
