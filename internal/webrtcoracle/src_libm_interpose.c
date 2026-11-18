//go:build webrtc_src_oracle && !webrtc_oracle && !webrtc_oracle_stock

// Interpose the libm entry points the AEC3 sources reach so the source build
// uses exactly the Go implementations in internal/aec3/cmath.go (host libm
// float results are not correctly rounded and differ between libms). How
// clang lowers std::pow(2.0f, x) and powf(10.f, x) depends on the target
// libm: exp2f everywhere, __exp10f on Darwin, exp10f or powf on glibc. Every
// spelling is covered; powf with another base aborts, since no Go helper
// mirrors it. src_main_test.go checks that the test binary imports none of
// these from libm. The webrtc_oracle_stock tag leaves them out, to measure
// goaec3 against the source build on the host libm
// (TestEc3StockLibmDivergence).
#include <stdio.h>
#include <stdlib.h>

float goaec3Exp2f(float);
float goaec3Exp10f(float);
float goaec3Log10f(float);

float exp2f(float x) { return goaec3Exp2f(x); }
float __exp10f(float x) { return goaec3Exp10f(x); }
float exp10f(float x) { return goaec3Exp10f(x); }
float log10f(float x) { return goaec3Log10f(x); }

// SincResampler builds its kernel with double sin/cos; AudioBuffer only
// resamples when its rates differ, which the oracle never configures.
double sin(double x) {
  fprintf(stderr, "goaec3 src oracle: unexpected sin(%a)\n", x);
  abort();
}
double cos(double x) {
  fprintf(stderr, "goaec3 src oracle: unexpected cos(%a)\n", x);
  abort();
}

float powf(float base, float x) {
  if (base == 2.0f) return goaec3Exp2f(x);
  if (base == 10.0f) return goaec3Exp10f(x);
  fprintf(stderr, "goaec3 src oracle: powf(%a, %a) has no Go counterpart\n", base, x);
  abort();
}
