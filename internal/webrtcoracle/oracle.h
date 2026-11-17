#ifndef GOAEC3_ORACLE_H
#define GOAEC3_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif
void oracle_ooura_fft(float* a, int inverse, int sse2);
void oracle_ooura_fft_scalar(float* a, int inverse);
#ifdef __cplusplus
}
#endif
#endif
