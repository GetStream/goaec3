//go:build webrtc_oracle

// Interposes log10f (imported by echo_remover.o and echo_remover_metrics.o)
// so the oracle uses the same correctly rounded log10f as the Go port
// (erLog10f in internal/aec3/echo_remover_metrics.go). The host log10f is not
// correctly rounded (log10f(1.1627032f) is one ulp off).
float goaec3EremLog10f(float);
float log10f(float x) { return goaec3EremLog10f(x); }
