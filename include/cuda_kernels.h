#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cstddef> // for size_t
#include <cfloat> // for FLOATMAX
#include <limits> // for min/max
#include <math.h> // for eigenvalues
#include <cusolverDn.h> // for svd
#include <cuda_runtime.h> // for svd
#include <stdexcept> // for svd
#include <iostream> // for svd

#ifdef USE_CUDA
void launch_cuda_add(const float* a, const float* b, float* result, size_t size);
void launch_cuda_dot(const float* a, const float* b, float* result, size_t size);
void launch_cuda_conv1d(const float* input, const float* kernel, float* output,
    int batch_size, int in_channels, int length,
    int kernel_size, int out_channels,
    int stride, int pad);
void launch_cuda_conv2d(const float* input, const float* kernel, float* output,
                         int x, int y, int z, int a, int b, int k, int stride, int pad);
void launch_cuda_conv3d(const float* input, const float* kernel, float* output,
                        int batch_size, int in_channels, int depth, int height, int width,
                        int kernel_depth, int kernel_height, int kernel_width, int out_channels,
                        int stride, int pad_depth, int pad_height, int pad_width);
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
void launch_cuda_argmax(const float* d_A, int* d_result, int axis, int dim0, int dim1);
void launch_cuda_argmin(const float* d_A, int* d_result, int axis, int dim0, int dim1);
void launch_cuda_matmul(const float* A, const float* B, float* C, int m, int n, int p);
void launch_cuda_transpose(const float* input, float* output, int m, int n);
void launch_cuda_greater_than_scalar(const float* input, float* output, float scalar, size_t size);
void launch_cuda_greater_than_tensor(const float* input1, const float* input2, float* output, size_t size);
void launch_cuda_maxpool2d(const float* input, float* output,
                           int batch_size, int channels, int height, int width,
                           int kernel_height, int kernel_width,
                           int stride, int pad_height, int pad_width);

void launch_cuda_avgpool2d(const float* input, float* output,
                           int batch_size, int channels, int height, int width,
                           int kernel_height, int kernel_width,
                           int stride, int pad_height, int pad_width);
void launch_cuda_inv(float* d_A, float* d_I, int size);
void launch_cuda_min(const float* d_A, float* d_result, int axis, int dim0, int dim1);
void launch_cuda_max(const float* d_A, float* d_result, int axis, int dim0, int dim1);
void launch_cuda_transpose(const float* d_A, float* d_result, int rows, int cols);
void launch_cuda_det(float* d_A, float* d_result, int n);
void launch_cuda_fill(float* data, float value, int n); // for eigen
void launch_cuda_matvec_mul(const float* matrix, const float* vector, float* result, int n); // for eigen
void launch_cuda_normalize(float* vector, float* norm, int n); // for eigen
void launch_cuda_svd(const float* d_A, float* d_U, float* d_S, float* d_VT, int m, int n);
void launch_cuda_reshape(const float* input, float* output, size_t total_size);
void launch_cuda_flatten(const float* input, float* output, size_t total_size);
void launch_cuda_expand_dims(const float* input, float* output, size_t total_size);
void launch_cuda_squeeze(const float* input, float* output, size_t total_size, size_t* new_shape, size_t* old_shape);
void launch_cuda_stack(const float* input, float* output, size_t total_size, size_t* new_shape, 
                       size_t* old_shape, int axis, size_t tensor_size);
void launch_cuda_concat(const float* A, const float* B, float* result, size_t size1, 
                        size_t size2, int axis, int dimA, int dimB);
void launch_cuda_permute(const float* input, float* output, const int* shape, const int* new_order, 
                         int num_dims, size_t size);
void launch_cuda_repeat(const float* input, float* output, const int* input_shape, const int* output_shape, 
                        int num_dims, int repeat_dim, size_t size);
void launch_cuda_equal(const float* a, const float* b, float* result, size_t size);
void launch_cuda_maxpool(const float* input, float* output, int batch_size, int channels, int length, 
                         int kernel_size, int stride, int pad, int output_length);
void launch_cuda_avgpool(const float* input, float* output, int batch_size, int channels, int length, 
                         int kernel_size, int stride, int pad, int output_length);
void launch_cuda_masked_assign(float* data, const float* mask, float value, size_t size);
#endif

#endif // CUDA_KERNELS_H
