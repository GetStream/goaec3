//go:build webrtc_oracle

#include "oracle.h"
#include "common_audio/third_party/ooura/fft_size_128/ooura_fft.h"
extern "C" void oracle_ooura_fft(float* a, int inverse, int sse2) {
  webrtc::OouraFft fft(sse2 != 0);
  if (inverse) fft.InverseFft(a); else fft.Fft(a);
}
