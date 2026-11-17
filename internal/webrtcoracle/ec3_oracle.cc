//go:build webrtc_oracle

// Shims over the prebuilt libwebrtc BlockProcessor and EchoCanceller3.
#pragma clang fp contract(off)
#pragma clang diagnostic ignored "-Wnullability-completeness"

#define EC3_FN(name) ec3_##name
#include "ec3_shim.inc"

extern "C" int ec3_config_num_fields(void) {
  int n = 0;
#define EC3_F(x) ++n;
#include "ec3_config_fields.inc"
#undef EC3_F
  return n;
}
