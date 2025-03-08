#include "cuda_kernels.h"

__global__ void cuda_add(const float* a, const float* b, float alpha, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + alpha * b[idx];
    }
}

void launch_cuda_add(const float* a, const float* b, float alpha, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_add<<<blocks, threads>>>(a, b, alpha, result, size);
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

__global__ void cuda_subtract(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

__global__ void cuda_add_scaled(const float* a, const float* b, float alpha, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + alpha * b[idx];
    }
}

__global__ void cuda_multiply(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void cuda_divide(const float* a, const float* b, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (b[idx] == 0) result[idx] = 0; // Handle division by zero
        else result[idx] = a[idx] / b[idx];
    }
}

__global__ void cuda_multiply_scalar(const float* a, float scalar, float* result, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
    }
}

void launch_cuda_subtract(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_subtract<<<blocks, threads>>>(a, b, result, size);
}

void launch_cuda_add_scaled(const float* a, const float* b, float alpha, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_add_scaled<<<blocks, threads>>>(a, b, alpha, result, size);
}

void launch_cuda_multiply(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_multiply<<<blocks, threads>>>(a, b, result, size);
}

void launch_cuda_divide(const float* a, const float* b, float* result, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    cuda_divide<<<blocks, threads>>>(a, b, result, size);
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

