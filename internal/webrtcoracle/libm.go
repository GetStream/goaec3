//go:build webrtc_oracle

package webrtcoracle

// The C definitions live in libm_interpose.c.

import "C"
import "github.com/GetStream/goaec3/internal/aec3"

//export goaec3Exp2f
func goaec3Exp2f(x C.float) C.float { return C.float(aec3.Exp2f(float32(x))) }

//export goaec3Exp10f
func goaec3Exp10f(x C.float) C.float { return C.float(aec3.Exp10f(float32(x))) }
