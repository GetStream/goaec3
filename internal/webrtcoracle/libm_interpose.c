//go:build webrtc_oracle && !webrtc_oracle_stock

// Interpose the libm entry points that libwebrtc AEC3 objects import so the
// oracle uses exactly the Go implementations in internal/aec3/cmath.go. The
// webrtc_oracle_stock tag leaves them out, to measure goaec3 against
// libwebrtc on the host libm (TestEc3StockLibmDivergence).
float goaec3Exp2f(float);
float goaec3Exp10f(float);
float goaec3Log10f(float);
float exp2f(float x) { return goaec3Exp2f(x); }
float __exp10f(float x) { return goaec3Exp10f(x); }
float log10f(float x) { return goaec3Log10f(x); }
