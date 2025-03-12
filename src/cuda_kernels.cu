#include "cuda_kernels.h"
#include <stdio.h>

__global__ void cuda_scalled_add(const float* a, const float* b, float alpha, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + alpha * b[idx];
    }
}

void launch_cuda_scalled_add(const float* a, const float* b, float alpha, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_scalled_add<<<blocks, threads>>>(a, b, alpha, result, size);
}

__global__ void cuda_dot(const float* a, const float* b, float* result, size_t size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = (idx < size) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    // Reduce within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, shared_mem[0]);
    }
}

void launch_cuda_dot(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_dot<<<blocks, threads, threads * sizeof(float)>>>(a, b, result, size);
}

__global__ void cuda_conv1d(const float* input, const float* kernel, float* output,
                            int batch_size, int in_channels, int length,
                            int kernel_size, int out_channels,
                            int stride, int pad) {
    int output_length = (length - kernel_size + 2 * pad) / stride + 1;

    int batch = blockIdx.x;
    int oc = blockIdx.y;
    int ol = threadIdx.x;

    if (batch < batch_size && oc < out_channels && ol < output_length) {
        float sum = 0.0f;

        for (int ks = 0; ks < kernel_size; ++ks) {
            for (int ic = 0; ic < in_channels; ++ic) {
                int input_pos = ol * stride + ks - pad;

                if (input_pos >= 0 && input_pos < length) {
                    float input_val = input[batch * in_channels * length + ic * length + input_pos];
                    float kernel_val = kernel[ks * in_channels * out_channels + ic * out_channels + oc];
                    sum += input_val * kernel_val;
                }
            }
        }

        output[batch * out_channels * output_length + oc * output_length + ol] = sum;
    }
}

void launch_cuda_conv1d(const float* input, const float* kernel, float* output,
                        int batch_size, int in_channels, int length,
                        int kernel_size, int out_channels,
                        int stride, int pad) {
    int output_length = (length - kernel_size + 2 * pad) / stride + 1;

    dim3 blocks(batch_size, out_channels);
    dim3 threads(output_length);

    cuda_conv1d<<<blocks, threads>>>(input, kernel, output, batch_size, in_channels, length, kernel_size, out_channels, stride, pad);
}

__global__ void cuda_conv2d(const float* input, const float* kernel, float* output,
                            int x, int y, int z, int a, int b, int k, int stride, int pad) {
    int out_x = (x - a + 2 * pad) / stride + 1;
    int out_y = (y - b + 2 * pad) / stride + 1;

    int filter = blockIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < out_x && j < out_y) {
        float sum = 0.0f;

        for (int di = 0; di < a; ++di) {
            for (int dj = 0; dj < b; ++dj) {
                for (int dz = 0; dz < z; ++dz) {
                    int input_i = i * stride + di - pad;
                    int input_j = j * stride + dj - pad;

                    if (input_i >= 0 && input_i < x && input_j >= 0 && input_j < y) {
                        float input_val = input[input_i * y * z + input_j * z + dz];
                        float kernel_val = kernel[di * b * z * k + dj * z * k + dz * k + filter];
                        sum += input_val * kernel_val;
                    }
                }
            }
        }

        output[i * out_y * k + j * k + filter] = sum;
    }
}

void launch_cuda_conv2d(const float* input, const float* kernel, float* output,
                        int x, int y, int z, int a, int b, int k, int stride, int pad) {
    int out_x = (x - a + 2 * pad) / stride + 1;
    int out_y = (y - b + 2 * pad) / stride + 1;

    dim3 blocks(k, (out_x + 15) / 16, (out_y + 15) / 16);
    dim3 threads(1, 16, 16);

    cuda_conv2d<<<blocks, threads>>>(input, kernel, output, x, y, z, a, b, k, stride, pad);
}

__global__ void cuda_conv3d(const float* input, const float* kernel, float* output,
                            int batch_size, int in_channels, int depth, int height, int width,
                            int kernel_depth, int kernel_height, int kernel_width, int out_channels,
                            int stride, int pad_depth, int pad_height, int pad_width) {
    int output_depth = (depth - kernel_depth + 2 * pad_depth) / stride + 1;
    int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
    int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

    int batch = blockIdx.x;
    int oc = blockIdx.y;
    int od = blockIdx.z;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    if (batch < batch_size && oc < out_channels && od < output_depth && oh < output_height && ow < output_width) {
        float sum = 0.0f;

        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kh = 0; kh < kernel_height; ++kh) {
                for (int kw = 0; kw < kernel_width; ++kw) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int input_d = od * stride + kd - pad_depth;
                        int input_h = oh * stride + kh - pad_height;
                        int input_w = ow * stride + kw - pad_width;

                        if (input_d >= 0 && input_d < depth &&
                            input_h >= 0 && input_h < height &&
                            input_w >= 0 && input_w < width) {
                            float input_val = input[batch * in_channels * depth * height * width +
                                                   ic * depth * height * width +
                                                   input_d * height * width +
                                                   input_h * width +
                                                   input_w];
                            float kernel_val = kernel[kd * kernel_height * kernel_width * in_channels * out_channels +
                                                     kh * kernel_width * in_channels * out_channels +
                                                     kw * in_channels * out_channels +
                                                     ic * out_channels +
                                                     oc];
                            sum += input_val * kernel_val;
                        }
                    }
                }
            }
        }

        output[batch * out_channels * output_depth * output_height * output_width +
               oc * output_depth * output_height * output_width +
               od * output_height * output_width +
               oh * output_width +
               ow] = sum;
    }
}

void launch_cuda_conv3d(const float* input, const float* kernel, float* output,
                        int batch_size, int in_channels, int depth, int height, int width,
                        int kernel_depth, int kernel_height, int kernel_width, int out_channels,
                        int stride, int pad_depth, int pad_height, int pad_width) {
    int output_depth = (depth - kernel_depth + 2 * pad_depth) / stride + 1;
    int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
    int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

    dim3 blocks(batch_size, out_channels, output_depth);
    dim3 threads(output_width, output_height);

    cuda_conv3d<<<blocks, threads>>>(input, kernel, output, batch_size, in_channels, depth, height, width,
                                     kernel_depth, kernel_height, kernel_width, out_channels, stride, pad_depth, pad_height, pad_width);
}

__global__ void cuda_power(const float* input, float* output, float exponent, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = powf(input[idx], exponent);
    }
}

void launch_cuda_power(const float* input, float* output, float exponent, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_power<<<blocks, threads>>>(input, output, exponent, size);
}


__global__ void cuda_add(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

void launch_cuda_add(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads -1) / threads;
    cuda_add<<<blocks, threads>>>(a, b, result, size);
}

__global__ void cuda_subtract(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

void launch_cuda_subtract(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_subtract<<<blocks, threads>>>(a, b, result, size);
}

__global__ void cuda_add_scaled(const float* a, const float* b, float alpha, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + alpha * b[idx];
    }
}

void launch_cuda_add_scaled(const float* a, const float* b, float alpha, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_add_scaled<<<blocks, threads>>>(a, b, alpha, result, size);
}

__global__ void cuda_multiply(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

void launch_cuda_multiply(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_multiply<<<blocks, threads>>>(a, b, result, size);
}

__global__ void cuda_divide(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (b[idx] == 0) result[idx] = 0; // Handle division by zero
        else result[idx] = a[idx] / b[idx];
    }
}

void launch_cuda_divide(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_divide<<<blocks, threads>>>(a, b, result, size);
}

__global__ void cuda_multiply_scalar(const float* a, float scalar, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
    }
}

void launch_cuda_multiply_scalar(const float* a, float scalar, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_multiply_scalar<<<blocks, threads>>>(a, scalar, result, size);
}

__global__ void cuda_sum(const float* input, float* output, int axis, size_t stride, size_t axis_size, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        size_t output_idx = (idx / (stride * axis_size)) * stride + (idx % stride);
        atomicAdd(&output[output_idx], input[idx]);
    }
}

void launch_cuda_sum(const float* input, float* output, int axis, size_t stride, size_t axis_size, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_sum<<<blocks, threads>>>(input, output, axis, stride, axis_size, size);
}

__global__ void cuda_mean(float* data, size_t size, float axis_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] /= axis_size;
    }
}

void launch_cuda_mean(float* data, size_t size, float axis_size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_mean<<<blocks, threads>>>(data, size, axis_size);
}

__global__ void cuda_matmul(const float* A, const float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void launch_cuda_matmul(const float* A, const float* B, float* C, int m, int n, int p) {
    dim3 threads(16, 16);
    dim3 blocks((p + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    cuda_matmul<<<blocks, threads>>>(A, B, C, m, n, p);
}

__global__ void cuda_greater_than_scalar(const float* input, float* output, float scalar, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > scalar) ? 1.0f : 0.0f;
    }
}

void launch_cuda_greater_than_scalar(const float* input, float* output, float scalar, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_greater_than_scalar<<<blocks, threads>>>(input, output, scalar, size);
}

__global__ void cuda_greater_than_tensor(const float* input1, const float* input2, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input1[idx] > input2[idx]) ? 1.0f : 0.0f;
    }
}

void launch_cuda_greater_than_tensor(const float* input1, const float* input2, float* output, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_greater_than_tensor<<<blocks, threads>>>(input1, input2, output, size);
}

__global__ void cuda_maxpool2d(const float* input, float* output,
                               int batch_size, int channels, int height, int width,
                               int kernel_height, int kernel_width,
                               int stride, int pad_height, int pad_width,
                               int output_height, int output_width) {
    int batch = blockIdx.x;
    int ch = blockIdx.y;
    int oh = blockIdx.z * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;

    if (batch < batch_size && ch < channels && oh < output_height && ow < output_width) {
        float max_val = -FLT_MAX;

        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int input_h = oh * stride + kh - pad_height;
                int input_w = ow * stride + kw - pad_width;

                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    float val = input[batch * channels * height * width +
                                    ch * height * width +
                                    input_h * width +
                                    input_w];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }

        output[batch * channels * output_height * output_width +
               ch * output_height * output_width +
               oh * output_width +
               ow] = max_val;
    }
}

void launch_cuda_maxpool2d(const float* input, float* output,
                           int batch_size, int channels, int height, int width,
                           int kernel_height, int kernel_width,
                           int stride, int pad_height, int pad_width) {
    int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
    int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

    dim3 blocks(batch_size, channels, (output_height + 15) / 16);
    dim3 threads(output_width, 16);

    cuda_maxpool2d<<<blocks, threads>>>(input, output, batch_size, channels, height, width,
                                        kernel_height, kernel_width, stride, pad_height, pad_width,
                                        output_height, output_width);
}

__global__ void cuda_avgpool2d(const float* input, float* output,
                               int batch_size, int channels, int height, int width,
                               int kernel_height, int kernel_width,
                               int stride, int pad_height, int pad_width,
                               int output_height, int output_width) {
    int batch = blockIdx.x;
    int ch = blockIdx.y;
    int oh = blockIdx.z * blockDim.y + threadIdx.y;
    int ow = threadIdx.x;

    if (batch < batch_size && ch < channels && oh < output_height && ow < output_width) {
        float sum = 0.0f;
        int count = 0;

        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int input_h = oh * stride + kh - pad_height;
                int input_w = ow * stride + kw - pad_width;

                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                    float val = input[batch * channels * height * width +
                                      ch * height * width +
                                      input_h * width +
                                      input_w];
                    sum += val;
                    count++;
                }
            }
        }

        output[batch * channels * output_height * output_width +
               ch * output_height * output_width +
               oh * output_width +
               ow] = sum / count;
    }
}

void launch_cuda_avgpool2d(const float* input, float* output,
                           int batch_size, int channels, int height, int width,
                           int kernel_height, int kernel_width,
                           int stride, int pad_height, int pad_width) {
    int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
    int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

    dim3 blocks(batch_size, channels, (output_height + 15) / 16);
    dim3 threads(output_width, 16);

    cuda_avgpool2d<<<blocks, threads>>>(input, output, batch_size, channels, height, width,
                                        kernel_height, kernel_width, stride, pad_height, pad_width,
                                        output_height, output_width);
}

__global__ void cuda_inv(float* A, float* I, int size) {
    //int col = threadIdx.x; // Column index
    int row = blockIdx.x;  // Row index

    for (int k = 0; k < size; k++) {
        float pivot = A[k * size + k];

        __syncthreads(); // Synchronize threads to ensure pivot is read correctly

        if (pivot == 0) return; // Singular matrix check (should be handled better)

        float scale = A[row * size + k] / pivot;

        if (row != k) {
            for (int j = 0; j < size; j++) {
                A[row * size + j] -= scale * A[k * size + j];
                I[row * size + j] -= scale * I[k * size + j];
            }
        }

        __syncthreads();
    }

    // Normalize diagonal elements
    float diag = A[row * size + row];
    for (int j = 0; j < size; j++) {
        A[row * size + j] /= diag;
        I[row * size + j] /= diag;
    }
}

void launch_cuda_inv(float* d_A, float* d_I, int size) {
    dim3 threads(size);  // N threads per row
    dim3 blocks(size);   // N blocks for N rows

    cuda_inv<<<blocks, threads>>>(d_A, d_I, size);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_min(const float* A, float* result, int axis, int dim0, int dim1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (axis == 0) { // Min along rows
        if (idx >= dim1) return;

        float min_val = A[idx];  // Start with first row value
        for (int i = 1; i < dim0; i++) {
            min_val = fminf(min_val, A[i * dim1 + idx]);
        }
        result[idx] = min_val;
    } 
    else if (axis == 1) { // Min along columns
        if (idx >= dim0) return;

        float min_val = A[idx * dim1];  // Start with first column value
        for (int j = 1; j < dim1; j++) {
            min_val = fminf(min_val, A[idx * dim1 + j]);
        }
        result[idx] = min_val;
    }
}

void launch_cuda_min(const float* d_A, float* d_result, int axis, int dim0, int dim1) {
    int threads = 256;
    int blocks = (axis == 0) ? (dim1 + threads - 1) / threads : (dim0 + threads - 1) / threads;

    cuda_min<<<blocks, threads>>>(d_A, d_result, axis, dim0, dim1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

#include <stdio.h>
#include <limits>

__global__ void cuda_max(const float* A, float* result, int axis, int dim0, int dim1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (axis == 0) { // Max along rows
        if (idx >= dim1) return;

        float max_val = A[idx];  // Start with first row value
        for (int i = 1; i < dim0; i++) {
            max_val = fmaxf(max_val, A[i * dim1 + idx]);
        }
        result[idx] = max_val;
    } 
    else if (axis == 1) { // Max along columns
        if (idx >= dim0) return;

        float max_val = A[idx * dim1];  // Start with first column value
        for (int j = 1; j < dim1; j++) {
            max_val = fmaxf(max_val, A[idx * dim1 + j]);
        }
        result[idx] = max_val;
    }
}

void launch_cuda_max(const float* d_A, float* d_result, int axis, int dim0, int dim1) {
    int threads = 256;
    int blocks = (axis == 0) ? (dim1 + threads - 1) / threads : (dim0 + threads - 1) / threads;

    cuda_max<<<blocks, threads>>>(d_A, d_result, axis, dim0, dim1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_argmax(const float* A, int* result, int axis, int dim0, int dim1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (axis == 0) { // Argmax along rows (output has dim1 elements)
        if (idx >= dim1) return;

        float max_val = A[idx];  
        int max_idx = 0;

        for (int i = 1; i < dim0; i++) {
            float val = A[i * dim1 + idx];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        result[idx] = max_idx;
    } 
    else if (axis == 1) { // Argmax along columns (output has dim0 elements)
        if (idx >= dim0) return;

        float max_val = A[idx * dim1];
        int max_idx = 0;

        for (int j = 1; j < dim1; j++) {
            float val = A[idx * dim1 + j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        result[idx] = max_idx;
    }
}

void launch_cuda_argmax(const float* d_A, int* d_result, int axis, int dim0, int dim1) {
    int threads = 256;
    int blocks = (axis == 0) ? (dim1 + threads - 1) / threads : (dim0 + threads - 1) / threads;

    cuda_argmax<<<blocks, threads>>>(d_A, d_result, axis, dim0, dim1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_argmin(const float* A, int* result, int axis, int dim0, int dim1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (axis == 0) { // Argmin along rows (output has dim1 elements)
        if (idx >= dim1) return;

        float min_val = A[idx];  
        int min_idx = 0;

        for (int i = 1; i < dim0; i++) {
            float val = A[i * dim1 + idx];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        result[idx] = min_idx;
    } 
    else if (axis == 1) { // Argmin along columns (output has dim0 elements)
        if (idx >= dim0) return;

        float min_val = A[idx * dim1];
        int min_idx = 0;

        for (int j = 1; j < dim1; j++) {
            float val = A[idx * dim1 + j];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        result[idx] = min_idx;
    }
}

void launch_cuda_argmin(const float* d_A, int* d_result, int axis, int dim0, int dim1) {
    int threads = 256;
    int blocks = (axis == 0) ? (dim1 + threads - 1) / threads : (dim0 + threads - 1) / threads;

    cuda_argmin<<<blocks, threads>>>(d_A, d_result, axis, dim0, dim1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_transpose(const float* A, float* result, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        result[x * rows + y] = A[y * cols + x];
    }
}

void launch_cuda_transpose(const float* d_A, float* d_result, int rows, int cols) {
    dim3 threads(16, 16);  
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    cuda_transpose<<<blocks, threads>>>(d_A, d_result, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_det(float* A, float* result, int n) {
    int tid = threadIdx.x;

    __shared__ float det; 
    if (tid == 0) det = 1.0f;  
    __syncthreads();

    for (int k = 0; k < n; k++) {
        if (tid == 0 && A[k * n + k] == 0.0f) {
            result[0] = 0.0f;
            return;
        }
        float pivot = A[k * n + k];
        if (tid == 0) det *= pivot;
        __syncthreads();

        for (int i = tid + k + 1; i < n; i += blockDim.x) {
            float factor = A[i * n + k] / pivot;
            for (int j = k; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
        }
        __syncthreads();
    }

    if (tid == 0) result[0] = det;
}

void launch_cuda_det(float* d_A, float* d_result, int n) {
    cuda_det<<<1, 32>>>(d_A, d_result, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_matmul_vec(const float* A, const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += A[idx * n + j] * x[j];
    }
    y[idx] = sum;
}

// CODE FOR EIGEN VALUES/VECTORS
__global__ void cuda_matvec_mul(const float* matrix, const float* vector, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += matrix[idx * n + i] * vector[i];
        }
        result[idx] = sum;
    }
}

__global__ void cuda_normalize(float* vector, float* norm, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = vector[idx] * vector[idx];
        atomicAdd(norm, val);
    }
    __syncthreads();
    if (idx < n) {
        vector[idx] /= sqrtf(*norm);
    }
}

void launch_cuda_matvec_mul(const float* matrix, const float* vector, float* result, int n) {
    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);

    cuda_matvec_mul<<<blocks, threads>>>(matrix, vector, result, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();  // Ensure kernel completes before proceeding
}

void launch_cuda_normalize(float* vector, float* norm, int n) {
    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);

    cudaMemset(norm, 0, sizeof(float));  // Reset norm to 0
    cuda_normalize<<<blocks, threads>>>(vector, norm, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();  // Ensure kernel completes before proceeding
}
// END OF CODE FOR EIGEN VALUES/VECTORS

void launch_cuda_svd(const float* d_A, float* d_U, float* d_S, float* d_VT, int m, int n) {
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    int lda = m;
    int ldu = m;
    int ldvt = n;
    int min_mn = std::min(m, n);

    float *d_work;
    int *devInfo;
    cudaMalloc(&d_work, sizeof(float) * (m * n));
    cudaMalloc(&devInfo, sizeof(int));

    int lwork;
    cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);
    cudaMalloc(&d_work, sizeof(float) * lwork);

    char jobu = 'A';  // Compute full U
    char jobvt = 'A'; // Compute full V^T
    cusolverDnSgesvd(cusolverH, jobu, jobvt, m, n, (float*)d_A, lda, d_S, d_U, ldu, d_VT, ldvt, d_work, lwork, nullptr, devInfo);

    int h_devInfo;
    cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_devInfo != 0) {
        std::cerr << "SVD failed with error: " << h_devInfo << std::endl;
        throw std::runtime_error("CUDA SVD computation failed");
    }

    // Cleanup
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
}

__global__ void cuda_reshape(const float* input, float* output, size_t total_size, size_t* new_shape, size_t current_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx]; // Copy the data
    }
}

void launch_cuda_reshape(const float* input, float* output, size_t total_size) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_reshape<<<blocks, threads>>>(input, output, total_size, nullptr, total_size);
}

__global__ void cuda_flatten(const float* input, float* output, size_t total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx]; // Copy the data
    }
}

void launch_cuda_flatten(const float* input, float* output, size_t total_size) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_flatten<<<blocks, threads>>>(input, output, total_size);
}
__global__ void cuda_expand_dims(const float* input, float* output, size_t total_size, size_t* new_shape, size_t current_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx]; // Copy the data to the new expanded tensor
    }
}

void launch_cuda_expand_dims(const float* input, float* output, size_t total_size) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_expand_dims<<<blocks, threads>>>(input, output, total_size, nullptr, total_size);
}

__global__ void cuda_squeeze(const float* input, float* output, size_t total_size, size_t* new_shape, size_t* old_shape) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx]; // Copy the data, skipping size 1 dimensions
    }
}

void launch_cuda_squeeze(const float* input, float* output, size_t total_size, size_t* new_shape, size_t* old_shape) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_squeeze<<<blocks, threads>>>(input, output, total_size, new_shape, old_shape);
}

__global__ void cuda_concat(const float* input1, const float* input2, float* output, size_t total_size, int axis, size_t* shape1, size_t* shape2, size_t* new_shape) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        // Compute the output index
        size_t stride = 1;
        //size_t axis_size = new_shape[axis];  // Get the axis size of the new shape

        // Determine the appropriate source tensor (input1 or input2)
        if (idx < shape1[axis] * stride) {
            // Copy data from input1
            output[idx] = input1[idx];
        } else {
            // Copy data from input2
            output[idx] = input2[idx - shape1[axis] * stride];
        }
    }
}

void launch_cuda_concat(const float* input1, const float* input2, float* output, size_t total_size, int axis, size_t* shape1, size_t* shape2, size_t* new_shape) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_concat<<<blocks, threads>>>(input1, input2, output, total_size, axis, shape1, shape2, new_shape);
}

__global__ void cuda_stack(const float* input, float* output, size_t total_size, size_t* new_shape, size_t* old_shape, int axis, size_t tensor_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_size) {
        // Compute the position in the output tensor
        // Determine which tensor the index corresponds to
        size_t tensor_idx = idx / tensor_size;
        size_t local_idx = idx % tensor_size;
        
        // Compute the corresponding index in the input tensor
        size_t input_idx = tensor_idx * tensor_size + local_idx;

        // Write the value to the corresponding place in the output tensor
        output[idx] = input[input_idx];
    }
}

void launch_cuda_stack(const float* input, float* output, size_t total_size, size_t* new_shape, size_t* old_shape, int axis, size_t tensor_size) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_stack<<<blocks, threads>>>(input, output, total_size, new_shape, old_shape, axis, tensor_size);
}

__global__ void cuda_permute(const float* input, float* output, size_t total_size, const size_t* new_shape, const size_t* old_shape, const int* new_order) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_size) {
        // Compute the corresponding indices for each dimension in the new order
        //size_t new_idx = idx;
        size_t input_idx = 0;

        for (int i = 0; i < old_shape[0]; ++i) {  // Loop over the number of dimensions
            int dim_idx = new_order[i];
            //size_t dimension_size = old_shape[dim_idx];
            size_t stride = 1;
            for (int j = dim_idx + 1; j < old_shape[0]; ++j) {
                stride *= old_shape[j];
            }

            size_t offset = idx / stride;
            idx = idx % stride;
            input_idx += offset * stride;
        }
        output[input_idx] = input[idx];
    }
}

void launch_cuda_permute(const float* input, float* output, size_t total_size, const size_t* new_shape, const size_t* old_shape, const int* new_order) {
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_permute<<<blocks, threads>>>(input, output, total_size, new_shape, old_shape, new_order);
}