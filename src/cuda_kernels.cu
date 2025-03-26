#include "cuda_kernels.h"
#include <stdio.h>

__global__ void setupIdentityKernel(float* matrix, int n);
__device__ float warpReduceSum(float val);

__device__ float atomicMinFloat(float* address, float value) {
    float old = *address, assumed;
    do {
        assumed = old;
        if (assumed <= value) break;  // Early exit if we already have a smaller value
        old = atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(value));
    } while (assumed != old);  // Keep looping until the atomic swap was successful
    return old;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void cuda_norm(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float temp = 0.0f;
    while (i < n) {
        temp += input[i] * input[i];
        i += gridDim.x * blockDim.x;
    }

    sdata[tid] = temp;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sqrtf(sdata[0]));
    }
}

void launch_cuda_norm(const float* input, float* output, int n) {
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 65535);
    cuda_norm<<<blocks, threads>>>(input, output, n);
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
        data[idx] /= axis_size; // ACTUALLY DIVIDE HERE
    }
}

void launch_cuda_mean(float* data, size_t size, float axis_size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_mean<<<blocks, threads>>>(data, size, axis_size);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); // Ensure kernel completes
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

__global__ void cuda_max(const float* data, float* result, size_t size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    shared_mem[tid] = (idx < size) ? data[idx] : -FLT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (idx + s) < size) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)result, __float_as_int(shared_mem[0]));
    }
}

void launch_cuda_max(const float* data, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    float min_value = -FLT_MAX;
    cudaMemcpy(d_result, &min_value, sizeof(float), cudaMemcpyHostToDevice);

    cuda_max<<<blocks, threads, threads * sizeof(float)>>>(data, d_result, size);

    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
}

__global__ void cuda_max_axis(const float* data, float* result, size_t outer_dim, size_t axis_size, size_t inner_stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= outer_dim * inner_stride) return;

    size_t outer_idx = tid / inner_stride;
    size_t inner_idx = tid % inner_stride;

    float max_val = -FLT_MAX;

    for (size_t i = 0; i < axis_size; ++i) {
        size_t idx = outer_idx * axis_size * inner_stride + i * inner_stride + inner_idx;
        max_val = fmaxf(max_val, data[idx]);
    }

    result[tid] = max_val;
}

void launch_cuda_max_axis(const float* data, float* result, size_t outer_dim, size_t axis_size, size_t inner_stride) {
    size_t total_threads = outer_dim * inner_stride;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    cuda_max_axis<<<blocks, threads>>>(data, result, outer_dim, axis_size, inner_stride);
}

__global__ void cuda_min(const float* d_A, float* d_result, size_t size) {
    __shared__ float shared_min[256];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    float local_min = FLT_MAX;
    while (tid < size) {
        local_min = fminf(local_min, d_A[tid]);
        tid += blockDim.x * gridDim.x;
    }

    shared_min[local_tid] = local_min;
    __syncthreads();

    // Reduce within a block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            shared_min[local_tid] = fminf(shared_min[local_tid], shared_min[local_tid + stride]);
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        atomicMinFloat(d_result, shared_min[0]);  // Atomically reduce across blocks
    }
}

void launch_cuda_min(const float* d_A, float* h_result, size_t size) {
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    float initial_min = FLT_MAX;
    cudaMemcpy(d_result, &initial_min, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_min<<<blocks, threads>>>(d_A, d_result, size);

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

__global__ void cuda_min_axis(const float* input, float* output, size_t outer_dim, size_t axis_size, size_t inner_stride) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= outer_dim * inner_stride) return;

    const size_t outer_idx = tid / inner_stride;
    const size_t inner_idx = tid % inner_stride;

    float min_val = INFINITY;  // Start with maximum possible value
    
    for (size_t a = 0; a < axis_size; ++a) {
        const size_t input_idx = outer_idx * axis_size * inner_stride 
                               + a * inner_stride 
                               + inner_idx;
        min_val = fminf(min_val, input[input_idx]);
    }
    
    output[tid] = min_val;
}

void launch_cuda_min_axis(const float* input, float* output, size_t outer_dim, size_t axis_size, size_t inner_stride) {
    const size_t total_elements = outer_dim * inner_stride;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    // Kernel handles initialization internally
    cuda_min_axis<<<blocks, threads>>>(input, output, outer_dim, axis_size, inner_stride);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA min_axis failed: ") + cudaGetErrorString(err)
        );
    }
    cudaDeviceSynchronize();
}

__global__ void cuda_argmax(const float* input, float* output, 
    const int* shape, int num_dims, int axis, 
    size_t outer_dim, size_t axis_size, size_t inner_stride) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= outer_dim * inner_stride) return;

    size_t outer_idx = tid / inner_stride;
    size_t inner_idx = tid % inner_stride;

    float max_val = -FLT_MAX;
    int max_index = 0;

    // Calculate strides for input tensor
    size_t axis_stride = 1;
    for (int i = axis + 1; i < num_dims; ++i) {
        axis_stride *= shape[i];
    }

    for (size_t a = 0; a < axis_size; ++a) {
        // Calculate input index using generalized N-dimensional indexing
        size_t input_idx = outer_idx * axis_size * axis_stride + 
        a * axis_stride + 
        inner_idx;

        float val = input[input_idx];
        if (val > max_val) {
            max_val = val;
            max_index = a;
        }
    }

    // Store result as float
    output[tid] = static_cast<float>(max_index);
}

void launch_cuda_argmax(const float* input, float* output, 
                        const int* shape, int num_dims, int axis,
                        size_t outer_dim, size_t axis_size, size_t inner_stride) {
    size_t total_threads = outer_dim * inner_stride;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    // Copy shape to device memory
    int* d_shape;
    cudaMalloc(&d_shape, num_dims * sizeof(int));
    cudaMemcpy(d_shape, shape, num_dims * sizeof(int), cudaMemcpyHostToDevice);

    cuda_argmax<<<blocks, threads>>>(input, output, d_shape, num_dims, axis, 
                                     outer_dim, axis_size, inner_stride);

    cudaFree(d_shape);
}

// In cuda_kernels.cu
__global__ void cuda_argmin(const float* input, float* output, 
                            const int* shape, int num_dims, int axis, 
                            size_t outer_dim, size_t axis_size, size_t inner_stride) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= outer_dim * inner_stride) return;

    size_t outer_idx = tid / inner_stride;
    size_t inner_idx = tid % inner_stride;

    float min_val = FLT_MAX;
    int min_index = 0;

    // Calculate strides for input tensor
    size_t axis_stride = 1;
    for (int i = axis + 1; i < num_dims; ++i) {
        axis_stride *= shape[i];
    }

    for (size_t a = 0; a < axis_size; ++a) {
        size_t input_idx = outer_idx * axis_size * axis_stride + 
        a * axis_stride + 
        inner_idx;

        float val = input[input_idx];
        if (val < min_val) {
            min_val = val;
            min_index = a;
        }
    }

    output[tid] = static_cast<float>(min_index); // Cast to float
}

void launch_cuda_argmin(const float* input, float* output, 
                        const int* shape, int num_dims, int axis,
                        size_t outer_dim, size_t axis_size, size_t inner_stride) {
    size_t total_threads = outer_dim * inner_stride;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;

    int* d_shape;
    cudaMalloc(&d_shape, num_dims * sizeof(int));
    cudaMemcpy(d_shape, shape, num_dims * sizeof(int), cudaMemcpyHostToDevice);

    cuda_argmin<<<blocks, threads>>>(input, output, d_shape, num_dims, axis, 
                                     outer_dim, axis_size, inner_stride);

    cudaFree(d_shape);
}

void launch_cuda_inv(float* d_A, float* d_invA, int n) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    // Initialize handles
    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);
    cudaStreamCreate(&stream);
    cusolverDnSetStream(cusolverH, stream);

    // Prepare device arrays
    int lwork = 0;
    int* d_ipiv = NULL;
    int* d_info = NULL;
    float* d_work = NULL;
    float** d_A_array = NULL;
    float** d_invA_array = NULL;

    const int batchSize = 1;  // Since we're processing a single matrix
    
    try {
        // Create array pointers for batched operation
        cudaMalloc(&d_A_array, batchSize * sizeof(float*));
        cudaMalloc(&d_invA_array, batchSize * sizeof(float*));
        cudaMemcpy(d_A_array, &d_A, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_invA_array, &d_invA, sizeof(float*), cudaMemcpyHostToDevice);

        // Query workspace size
        cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, n, &lwork);

        // Allocate device memory
        cudaMalloc(&d_ipiv, n * batchSize * sizeof(int));
        cudaMalloc(&d_info, batchSize * sizeof(int));
        cudaMalloc(&d_work, lwork * batchSize * sizeof(float));

        // Perform LU factorization
        cusolverDnSgetrf(cusolverH, n, n, d_A, n, d_work, d_ipiv, d_info);

        // Check singularity
        int info = 0;
        cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (info != 0) {
            throw std::runtime_error("Matrix is singular at position " + std::to_string(info));
        }

        // Prepare identity matrix for inversion
        float* d_identity;
        cudaMalloc(&d_identity, n * n * sizeof(float));
        cudaMemsetAsync(d_identity, 0, n * n * sizeof(float), stream);
        int num_elements = n * n;
        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        setupIdentityKernel<<<blocks, threads, 0, stream>>>(d_identity, n);
        //setupIdentityKernel<<<(n + 255)/256, 256, 0, stream>>>(d_identity, n);
        
        // Compute inverse using getrs
        cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, n, n, d_A, n, d_ipiv, d_identity, n, d_info);
        
        // Copy the result to output
        cudaMemcpyAsync(d_invA, d_identity, n * n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // Cleanup temporary identity matrix
        cudaFree(d_identity);

        // Check inversion success
        cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (info != 0) {
            throw std::runtime_error("Inversion failed with error code " + std::to_string(info));
        }

        cudaStreamSynchronize(stream);
    }
    catch (...) {
        // Cleanup and rethrow
        cudaFree(d_A_array);
        cudaFree(d_invA_array);
        cudaFree(d_ipiv);
        cudaFree(d_info);
        cudaFree(d_work);
        cusolverDnDestroy(cusolverH);
        cublasDestroy(cublasH);
        cudaStreamDestroy(stream);
        throw;
    }

    // Cleanup
    cudaFree(d_A_array);
    cudaFree(d_invA_array);
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);
    cudaStreamDestroy(stream);
}

__global__ void setupIdentityKernel(float* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx / n;  // Calculate row from linear index
        int col = idx % n;  // Calculate column from linear index
        matrix[row * n + col] = (row == col) ? 1.0f : 0.0f;
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

__global__ void cuda_fill(float* data, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

void launch_cuda_fill(float* data, float value, int n) {
    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);

    cuda_fill<<<blocks, threads>>>(data, value, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();  // Ensure kernel completes before proceeding
}

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
    __shared__ float s_norm;
    
    // First calculate squared sum
    float temp = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        temp += vector[i] * vector[i];
    }
    
    // Block-wise reduction
    temp = blockReduceSum(temp);
    
    if (threadIdx.x == 0) {
        s_norm = sqrtf(temp); // Store sqrt(sum) in shared memory
    }
    __syncthreads();

    // Normalize vector
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        vector[i] /= s_norm;
    }
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *norm = s_norm; // Store actual norm, not squared
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

__global__ void cuda_concat(const float* A, const float* B, float* result, size_t size1, size_t size2, int axis, int dimA, int dimB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size1) {
        result[idx] = A[idx];
    } else if (idx < size1 + size2) {
        result[idx] = B[idx - size1];
    }
}

void launch_cuda_concat(const float* d_A, const float* d_B, float* d_result, size_t size1, size_t size2, int axis, int dimA, int dimB) {
    size_t total_size = size1 + size2;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    cuda_concat<<<blocks, threads>>>(d_A, d_B, d_result, size1, size2, axis, dimA, dimB);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_permute(const float* input, float* output, const int* shape, const int* new_order, int num_dims, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Calculate the original multi-dimensional index
        int original_index = idx;
        int original_coords[8]; // Assuming max 8 dimensions
        for (int i = num_dims - 1; i >= 0; --i) {
            original_coords[i] = original_index % shape[i];
            original_index /= shape[i];
        }

        // Calculate the new index based on the new order
        int new_index = 0;
        int stride = 1;
        for (int i = num_dims - 1; i >= 0; --i) {
            new_index += original_coords[new_order[i]] * stride;
            stride *= shape[new_order[i]];
        }

        // Copy the value to the new position
        output[new_index] = input[idx];
    }
}

void launch_cuda_permute(const float* d_input, float* d_output, const int* shape, const int* new_order, int num_dims, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch kernel
    cuda_permute<<<blocks, threads>>>(d_input, d_output, shape, new_order, num_dims, size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

__global__ void cuda_repeat(const float* input, float* output, const int* input_shape, const int* output_shape, int num_dims, int repeat_dim, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // Debugging: Print num_dims and output_shape
        if (idx == 0) {
            printf("num_dims: %d\n", num_dims);
            printf("output_shape: ");
            for (int i = 0; i < num_dims; ++i) {
                printf("%d ", output_shape[i]);
            }
            printf("\n");
        }

        // Calculate the multi-dimensional index in the output tensor
        int output_index = idx;
        int output_coords[8]; // Assuming max 8 dimensions
        for (int i = num_dims - 1; i >= 0; --i) {
            output_coords[i] = output_index % output_shape[i];
            output_index /= output_shape[i];
        }

        // Debugging: Print output_coords
        if (idx == 0) {
            printf("output_coords: ");
            for (int i = 0; i < num_dims; ++i) {
                printf("%d ", output_coords[i]);
            }
            printf("\n");
        }

        // Calculate the corresponding index in the input tensor
        int input_index = 0;
        int stride = 1;
        for (int i = num_dims - 1; i >= 0; --i) {
            if (i == repeat_dim) {
                input_index += (output_coords[i] % input_shape[i]) * stride;
            } else {
                input_index += output_coords[i] * stride;
            }
            stride *= input_shape[i];
        }

        // Debugging: Print input_index
        if (idx == 0) {
            printf("input_index: %d\n", input_index);
        }

        // Copy the value from the input tensor to the output tensor
        output[idx] = input[input_index];
    }
}

void launch_cuda_repeat(const float* d_input, float* d_output, const int* input_shape, const int* output_shape, int num_dims, int repeat_dim, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Allocate device memory for shapes
    int* d_input_shape;
    int* d_output_shape;
    cudaMalloc(&d_input_shape, num_dims * sizeof(int));
    cudaMalloc(&d_output_shape, num_dims * sizeof(int));
    cudaMemcpy(d_input_shape, input_shape, num_dims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_shape, output_shape, num_dims * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    cuda_repeat<<<blocks, threads>>>(d_input, d_output, d_input_shape, d_output_shape, num_dims, repeat_dim, size);

    // Free device memory
    cudaFree(d_input_shape);
    cudaFree(d_output_shape);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch any errors
    cudaDeviceSynchronize();
}

__global__ void cuda_equal(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

void launch_cuda_equal(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_equal<<<blocks, threads>>>(a, b, result, size);
}

__global__ void cuda_maxpool(const float* input, float* output, int batch_size, int channels, int length, int kernel_size, int stride, int pad, int output_length) {
    int ol = blockIdx.x * blockDim.x + threadIdx.x; // Output position (1D)
    int ch = blockIdx.y * blockDim.y + threadIdx.y; // Channel
    int batch = blockIdx.z;                        // Batch

    if (batch < batch_size && ch < channels && ol < output_length) {
        float max_val = -FLT_MAX;

        for (int ks = 0; ks < kernel_size; ++ks) {
            int input_pos = ol * stride + ks - pad;

            if (input_pos >= 0 && input_pos < length) {
                float val = input[batch * channels * length + ch * length + input_pos];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }

        output[batch * channels * output_length + ch * output_length + ol] = max_val;
    }
}

void launch_cuda_maxpool(const float* input, float* output, int batch_size, int channels, int length, int kernel_size, int stride, int pad, int output_length) {
    dim3 threads(16, 16); // 16x16 threads per block
    dim3 blocks((output_length + threads.x - 1) / threads.x, 
                (channels + threads.y - 1) / threads.y, 
                batch_size);

    cuda_maxpool<<<blocks, threads>>>(input, output, batch_size, channels, length, kernel_size, stride, pad, output_length);
}

__global__ void cuda_avgpool(const float* input, float* output, int batch_size, int channels, int length, int kernel_size, int stride, int pad, int output_length) {
    int ol = blockIdx.x * blockDim.x + threadIdx.x; // Output position (1D)
    int ch = blockIdx.y * blockDim.y + threadIdx.y; // Channel
    int batch = blockIdx.z;                        // Batch

    if (batch < batch_size && ch < channels && ol < output_length) {
        float sum = 0.0f;
        int count = 0;

        for (int ks = 0; ks < kernel_size; ++ks) {
            int input_pos = ol * stride + ks - pad;

            if (input_pos >= 0 && input_pos < length) {
                float val = input[batch * channels * length + ch * length + input_pos];
                sum += val;
                count++;
            }
        }

        output[batch * channels * output_length + ch * output_length + ol] = sum / count;
    }
}

void launch_cuda_avgpool(const float* input, float* output, int batch_size, int channels, int length, int kernel_size, int stride, int pad, int output_length) {
    dim3 threads(16, 16); // 16x16 threads per block
    dim3 blocks((output_length + threads.x - 1) / threads.x, 
                (channels + threads.y - 1) / threads.y, 
                batch_size);

    cuda_avgpool<<<blocks, threads>>>(input, output, batch_size, channels, length, kernel_size, stride, pad, output_length);
}

// CUDA kernel for masked assignment
__global__ void cuda_masked_assign(float* data, const float* mask, float value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && mask[idx] != 0.0f) {
        data[idx] = value;
    }
}

// Function to launch the CUDA kernel
void launch_cuda_masked_assign(float* data, const float* mask, float value, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_masked_assign<<<blocks, threads>>>(data, mask, value, size);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}