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

