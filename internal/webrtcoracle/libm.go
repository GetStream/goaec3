//go:build webrtc_oracle || webrtc_src_oracle

package webrtcoracle

// The libm symbols that call these exports are defined in libm_interpose.c
// (prebuilt library) and src_libm_interpose.c (source build).

import "C"
import "github.com/GetStream/goaec3/internal/aec3"

//export goaec3Exp2f
func goaec3Exp2f(x C.float) C.float { return C.float(aec3.Exp2f(float32(x))) }

//export goaec3Exp10f
func goaec3Exp10f(x C.float) C.float { return C.float(aec3.Exp10f(float32(x))) }

//export goaec3Log10f
func goaec3Log10f(x C.float) C.float { return C.float(aec3.Log10f(float32(x))) }
