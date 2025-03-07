#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cstddef> // for size_t

#ifdef USE_CUDA
void launch_cuda_add(const float* a, const float* b, float alpha, float* result, size_t size);
void launch_cuda_dot(const float* a, const float* b, float* result, size_t size);
void launch_cuda_conv2d(const float* input, const float* kernel, float* output,
                         int x, int y, int z, int a, int b, int k, int stride, int pad);
void launch_cuda_power(const float* input, float* output, float exponent, size_t size);
void launch_cuda_subtract(const float* a, const float* b, float* result, size_t size);
void launch_cuda_add_scaled(const float* a, const float* b, float alpha, float* result, size_t size);
void launch_cuda_multiply(const float* a, const float* b, float* result, size_t size);
void launch_cuda_divide(const float* a, const float* b, float* result, size_t size);
void launch_cuda_multiply_scalar(const float* a, float scalar, float* result, size_t size);
void launch_cuda_sum(const float* input, float* output, int axis, size_t stride, size_t axis_size, size_t size);
void launch_cuda_mean(float* data, size_t size, float axis_size);
void launch_cuda_max(const float* input, float* output, int axis, size_t stride, size_t axis_size, size_t size);
void launch_cuda_min(const float* input, float* output, int axis, size_t stride, size_t axis_size, size_t size);
void launch_cuda_argmax(const float* input, int* output, int axis, size_t stride, size_t axis_size, size_t size);
void launch_cuda_argmin(const float* input, int* output, int axis, size_t stride, size_t axis_size, size_t size);
void launch_cuda_matmul(const float* A, const float* B, float* C, int m, int n, int p);
void launch_cuda_transpose(const float* input, float* output, int m, int n);
#endif

#endif // CUDA_KERNELS_H
