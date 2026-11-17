#ifndef GOAEC3_OOURA_KERN_ORACLE_H
#define GOAEC3_OOURA_KERN_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif
// opt is the webrtc::Aec3Optimization value (kNone=0, kSse2, kAvx2, kNeon).
void oracle_kern_spectrum(int opt, const float* re, const float* im, float* out);
void oracle_kern_sqrt(int opt, float* x, int n);
void oracle_kern_multiply(int opt, const float* x, const float* y, float* z, int n);
void oracle_kern_accumulate(int opt, const float* x, float* z, int n);
// x86 build only: OouraFft compiled from source with the SSE2 path.
void oracle_ooura_fft_x86(float* a, int inverse, int sse2);
#ifdef __cplusplus
}
#endif
#endif
