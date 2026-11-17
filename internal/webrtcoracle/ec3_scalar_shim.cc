//go:build webrtc_oracle

// BlockProcessor / EchoCanceller3 shims over the scalar source build.
#include "ec3_scalar_prelude.h"

#define EC3_FN(name) ec3s_##name
#include "ec3_shim.inc"
