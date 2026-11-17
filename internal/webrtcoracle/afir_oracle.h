#ifndef GOAEC3_AFIR_ORACLE_H
#define GOAEC3_AFIR_ORACLE_H
#ifdef __cplusplus
extern "C" {
#endif

// FftData is passed as 130 floats: re[65] then im[65]. Filters H are
// [partition][channel] arrays of FftData. lane: 0 = scalar, 1 = NEON.

typedef struct afir_rb afir_rb;
afir_rb* afir_rb_new(int size, int num_bands, int num_channels);
void afir_rb_free(afir_rb* rb);
void afir_rb_set_fft(afir_rb* rb, int index, int ch, const float* fft130);
void afir_rb_set_spectrum(afir_rb* rb, int index, int ch, const float* x65);
void afir_rb_set_block(afir_rb* rb, int index, int band, int ch, const float* x64);
void afir_rb_set_read(afir_rb* rb, int fft_read, int spectrum_read, int block_read);

void afir_apply_filter(afir_rb* rb, int lane, int num_partitions, const float* H, int h_partitions, int num_channels, float* S);
void afir_adapt_partitions(afir_rb* rb, int lane, const float* G, int num_partitions, float* H, int h_partitions, int num_channels);
void afir_freq_resp(int lane, int num_partitions, const float* H, int h_partitions, int num_channels, float* H2);
void afir_erl(int lane, const float* H2, int n, float* erl);

// AdaptiveFirFilter. scalar_fft selects the source-built class whose
// Aec3Fft runs the scalar Ooura FFT (the prebuilt one runs NEON Ooura).
typedef struct afir_filter afir_filter;
afir_filter* afir_filter_new(int scalar_fft, int max_size, int initial_size, int change_duration, int num_channels, int lane);
void afir_filter_free(afir_filter* f);
void afir_filter_filter(afir_filter* f, afir_rb* rb, float* S);
void afir_filter_adapt(afir_filter* f, afir_rb* rb, const float* G);
// ir has capacity GetTimeDomainLength(max_size); returns its new size.
int afir_filter_adapt_ir(afir_filter* f, afir_rb* rb, const float* G, float* ir, int ir_len);
int afir_filter_freq_resp(afir_filter* f, float* H2);
void afir_filter_set_size(afir_filter* f, int size, int immediate);
void afir_filter_handle_echo_path_change(afir_filter* f);
void afir_filter_scale(afir_filter* f, float factor);
void afir_filter_set_filter(afir_filter* f, int num_partitions, const float* H);
void afir_filter_get_filter(afir_filter* f, float* H);
int afir_filter_size(afir_filter* f);

typedef struct afir_rsa afir_rsa;
afir_rsa* afir_rsa_new(int refined_length_blocks);
void afir_rsa_free(afir_rsa* a);
void afir_rsa_update(afir_rsa* a, afir_rb* rb, int delay, int has_delay);
int afir_rsa_poor_signal_excitation(afir_rsa* a);
void afir_rsa_mask(afir_rsa* a, float* v65);
int afir_rsa_narrow_peak_band(afir_rsa* a, int* band);

typedef struct afir_coarse afir_coarse;
afir_coarse* afir_coarse_new(int length_blocks, float rate, float noise_gate, int change_duration);
void afir_coarse_free(afir_coarse* g);
void afir_coarse_handle_echo_path_change(afir_coarse* g);
void afir_coarse_set_config(afir_coarse* g, int length_blocks, float rate, float noise_gate, int immediate);
void afir_coarse_compute(afir_coarse* g, const float* X2, afir_rsa* a, const float* E130, int size_partitions, int saturated, float* G130);

typedef struct afir_refined afir_refined;
afir_refined* afir_refined_new(const float* cfg6, int change_duration);
void afir_refined_free(afir_refined* g);
void afir_refined_handle_echo_path_change(afir_refined* g, int gain_change, int delay_change, int clock_drift);
void afir_refined_set_config(afir_refined* g, const float* cfg6, int immediate);
void afir_refined_compute(afir_refined* g, const float* X2, afir_rsa* a, const float* E130, const float* E2_refined, const float* E2_coarse, const float* erl, int size_partitions, int saturated, int disallow_leakage_diverged, float* G130);

#ifdef __cplusplus
}
#endif
#endif
