//go:build webrtc_oracle

// Interpose the libm entry points that libwebrtc AEC3 objects import so the
// oracle uses exactly the Go implementations in internal/aec3/cmath.go.
float goaec3Exp2f(float);
float goaec3Exp10f(float);
float exp2f(float x) { return goaec3Exp2f(x); }
float __exp10f(float x) { return goaec3Exp10f(x); }
