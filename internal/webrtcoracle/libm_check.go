//go:build webrtc_oracle

package webrtcoracle

/*
#include <math.h>
static float goaec3_call_exp2f(float x) { return exp2f(x); }
*/
import "C"

func cExp2f(x float32) float32 { return float32(C.goaec3_call_exp2f(C.float(x))) }
