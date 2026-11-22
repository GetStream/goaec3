//go:build webrtc_oracle && !webrtc_oracle_stock

// Interposes the libm entry points that the AEC3 objects of libwebrtc import,
// so that the oracle uses exactly the Go implementations in
// internal/aec3/cmath.go (src_libm_interpose.c gives the reason). The
// webrtc_oracle_stock tag leaves them out, to measure goaec3 against
// libwebrtc on the host libm (TestEc3StockLibmDivergence).
float goaec3Exp2f(float);
float goaec3Exp10f(float);
float goaec3Log10f(float);
float exp2f(float x) { return goaec3Exp2f(x); }
float __exp10f(float x) { return goaec3Exp10f(x); }
float log10f(float x) { return goaec3Log10f(x); }
