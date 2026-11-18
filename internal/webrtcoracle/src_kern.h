#ifndef GOAEC3_SRC_KERN_H
#define GOAEC3_SRC_KERN_H
#ifdef __cplusplus
extern "C" {
#endif

// The lane arguments are Aec3Optimization values (0 scalar, 1 SSE2, 2 AVX2,
// 3 NEON). Every entry point returns 0, or -1 when this build does not
// contain the lane's kernel.

int src_kern_has_lane(int lane);

int src_kern_mf_core(int lane, int x_start_index, float x2_sum_threshold, float smoothing, const float* x,
                     int x_size, const float* y, int y_size, float* h, int h_size, int* filters_updated,
                     float* error_sum, int compute_accumulated_error, float* accumulated_error,
                     int accumulated_error_size, float* scratch, int scratch_size);

// A RenderBuffer over an FftBuffer of size x channels; only the FFT buffer
// and its read index are used by the adaptive filter kernels. FftData is
// passed as re[65] then im[65].
typedef struct src_kern_rb src_kern_rb;
src_kern_rb* src_kern_rb_new(int size, int num_channels);
void src_kern_rb_free(src_kern_rb* rb);
void src_kern_rb_set_fft(src_kern_rb* rb, int index, int ch, const float* fft130);
void src_kern_rb_set_read(src_kern_rb* rb, int read);

// H is [h_partitions][num_channels] FftData.
int src_kern_apply_filter(src_kern_rb* rb, int lane, int num_partitions, const float* H, int h_partitions,
                          int num_channels, float* S);
int src_kern_adapt_partitions(src_kern_rb* rb, int lane, const float* G, int num_partitions, float* H,
                              int h_partitions, int num_channels);
// H2 is [h_partitions][65], updated in place.
int src_kern_freq_resp(int lane, int num_partitions, const float* H, int h_partitions, int num_channels,
                       float* H2);
int src_kern_erl(int lane, const float* H2, int n, float* erl);

#ifdef __cplusplus
}
#endif
#endif
