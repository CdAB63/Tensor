#include "Tensor.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

Tensor::Tensor(const std::vector<int>& shape, bool use_gpu)
    : shape_(shape), use_gpu_(use_gpu) {
    allocate_memory();
}

Tensor::~Tensor() {
    free_memory();
}

void Tensor::load_data(const std::vector<float>& data) {
    if (data.size() != this->size()) {
        throw std::runtime_error("Data size does not match tensor size");
    }

    if (use_gpu_) {
#ifdef USE_CUDA
        cudaError_t err = cudaMemcpy(data_.get(), data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to load data to GPU: " + std::string(cudaGetErrorString(err)));
        }
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        std::copy(data.begin(), data.end(), data_.get());
    }
}

std::vector<float> Tensor::get_data() const {
    std::vector<float> result(this->size());

    if (use_gpu_) {
#ifdef USE_CUDA
        // Copy data from GPU to host
        cudaError_t err = cudaMemcpy(result.data(), data_.get(), result.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to get data from GPU: " + std::string(cudaGetErrorString(err)));
        }
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // Copy data from CPU
        std::copy(data_.get(), data_.get() + result.size(), result.begin());
    }

    return result;
}

void Tensor::allocate_memory() {
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        float* gpu_data;
        cudaError_t err = cudaMalloc(&gpu_data, size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA memory allocation failed");
        }
        data_ = std::shared_ptr<float>(gpu_data, [](float* ptr) { cudaFree(ptr); });
        //std::cout << "CUDA memory allocated successfully for tensor of size " << size << std::endl;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        data_ = std::shared_ptr<float>(new float[size], [](float* ptr) { delete[] ptr; });
    }
}

bool Tensor::use_gpu() const {
    return use_gpu_;
}

float Tensor::at(const std::vector<int>& indices) const {
    if (use_gpu_) {
        throw std::runtime_error("Element access not supported for GPU tensors. Use get_data() to copy to CPU first.");
    }
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("Number of indices does not match tensor rank");
    }
    size_t flat_index = 0;
    for (int i = 0; i < shape_.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds in dimension " + std::to_string(i));
        }
        flat_index = flat_index * shape_[i] + indices[i];
    }
    return data_.get()[flat_index];
}

Tensor Tensor::slice(const std::vector<std::pair<int, int>>& ranges) const {
    if (ranges.size() != shape_.size()) {
        throw std::runtime_error("Number of ranges must match tensor dimensions");
    }

    std::vector<int> new_shape;
    std::vector<int> starts;
    for (size_t i = 0; i < ranges.size(); ++i) {
        int start = ranges[i].first;
        int end = ranges[i].second;
        if (start < 0 || end > shape_[i] || start >= end) {
            throw std::invalid_argument("Invalid range for dimension " + std::to_string(i));
        }
        new_shape.push_back(end - start);
        starts.push_back(start);
    }

    Tensor result(new_shape, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        throw std::runtime_error("GPU slicing not implemented");
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        size_t total_elements = result.size();
        const float* src = data_.get();
        float* dest = result.data_.get();

        for (size_t idx = 0; idx < total_elements; ++idx) {
            std::vector<int> new_indices(new_shape.size());
            size_t temp = idx;
            for (int i = new_shape.size() - 1; i >= 0; --i) {
                new_indices[i] = temp % new_shape[i];
                temp /= new_shape[i];
            }

            size_t src_idx = 0;
            for (size_t i = 0; i < new_indices.size(); ++i) {
                src_idx = src_idx * shape_[i] + (starts[i] + new_indices[i]);
            }

            dest[idx] = src[src_idx];
        }
    }

    return result;
}

float Tensor::rayleigh_quotient(const Tensor& v) const {
    if (shape().size() != 2 || shape()[0] != shape()[1]) {
        throw std::runtime_error("Matrix must be square for Rayleigh quotient");
    }
    
    // Ensure vectors are properly flattened
    Tensor Av = this->matmul(v).flatten();
    Tensor v_flat = v.flatten();
    
    float numerator = Av.dot(v_flat);  // v^T A v
    float denominator = v_flat.dot(v_flat); // v^T v
    
    if (denominator == 0) throw std::runtime_error("Zero vector in Rayleigh quotient");
    
    return numerator / denominator;
}

void Tensor::free_memory() {
    // Memory is automatically managed by shared_ptr
}

Tensor Tensor::copy() const {
    Tensor new_tensor(shape_, use_gpu_);
    
    if (use_gpu_) {
#ifdef USE_CUDA
        size_t bytes = size() * sizeof(float);
        float* new_data;
        cudaMalloc(&new_data, bytes);
        cudaMemcpy(new_data, data_.get(), bytes, cudaMemcpyDeviceToDevice);
        new_tensor.data_ = std::shared_ptr<float>(new_data, [](float* p) { cudaFree(p); });
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        float* new_data = new float[size()];
        std::copy(data_.get(), data_.get() + size(), new_data);
        new_tensor.data_ = std::shared_ptr<float>(new_data, [](float* p) { delete[] p; });
    }
    
    return new_tensor;
}

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_add(data_.get(), other.data_.get(), result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            result.data_.get()[i] = data_.get()[i] + other.data_.get()[i];
        }
    }

    return result;
}

float Tensor::dot(const Tensor& other) const {
    if (shape_.size() != 1 || other.shape_.size() != 1) {
        std::cerr << "Error: Tensor::dot() was defined for vectors, otherwise it is matmul" << std::endl;
        throw std::runtime_error("Shape mismatch");
    }
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    float result = 0.0f;
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        cudaMemset(d_result, 0, sizeof(float));
        launch_cuda_dot(data_.get(), other.data_.get(), d_result, size);
        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            result += data_.get()[i] * other.data_.get()[i];
        }
    }

    return result;
}

Tensor Tensor::conv1d_cpu(const Tensor& kernel, int stride, bool padding) const {
    // Input dimensions
    int batch_size = shape_[0];
    int in_channels = shape_[1];
    int length = shape_[2];

    // Kernel dimensions
    int kernel_size = kernel.shape()[0];
    int out_channels = kernel.shape()[2];

    // Padding
    int pad = padding ? (kernel_size - 1) / 2 : 0;

    // Output dimensions
    int output_length = (length - kernel_size + 2 * pad) / stride + 1;

    // Create output tensor
    Tensor output({batch_size, out_channels, output_length}, use_gpu_);

    // Perform convolution
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int ol = 0; ol < output_length; ++ol) {
                float sum = 0.0f;

                for (int ks = 0; ks < kernel_size; ++ks) {
                    for (int ic = 0; ic < in_channels; ++ic) {
                        int input_pos = ol * stride + ks - pad;

                        if (input_pos >= 0 && input_pos < length) {
                            float input_val = data_.get()[batch * in_channels * length + ic * length + input_pos];
                            float kernel_val = kernel.data()[ks * in_channels * out_channels + ic * out_channels + oc];
                            sum += input_val * kernel_val;
                        }
                    }
                }

                output.data()[batch * out_channels * output_length + oc * output_length + ol] = sum;
            }
        }
    }

    return output;
}

Tensor Tensor::conv1d(const Tensor& kernel, int stride, bool padding) const {
    if (use_gpu_) {
#ifdef USE_CUDA
        int batch_size = shape_[0];
        int in_channels = shape_[1];
        int length = shape_[2];

        int kernel_size = kernel.shape()[0];
        int out_channels = kernel.shape()[2];

        int pad = padding ? (kernel_size - 1) / 2 : 0;

        int output_length = (length - kernel_size + 2 * pad) / stride + 1;

        Tensor output({batch_size, out_channels, output_length}, use_gpu_);

        launch_cuda_conv1d(data_.get(), kernel.data(), output.data(), batch_size, in_channels, length, kernel_size, out_channels, stride, pad);

        return output;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // Call the CPU implementation
        return conv1d_cpu(kernel, stride, padding);
    }
}

Tensor Tensor::conv2d_cpu(const Tensor& kernel, int stride, bool padding) const {
    // Input dimensions
    int x = shape_[0];
    int y = shape_[1];
    int z = shape_[2];

    // Kernel dimensions
    int a = kernel.shape()[0];
    int b = kernel.shape()[1];
    int k = kernel.shape()[3]; // Number of filters

    // Padding
    int pad = padding ? (a - 1) / 2 : 0;

    // Output dimensions
    int out_x = (x - a + 2 * pad) / stride + 1;
    int out_y = (y - b + 2 * pad) / stride + 1;

    // Create output tensor
    Tensor output({out_x, out_y, k}, use_gpu_);

    // Perform convolution
    for (int filter = 0; filter < k; ++filter) {
        for (int i = 0; i < out_x; ++i) {
            for (int j = 0; j < out_y; ++j) {
                float sum = 0.0f;

                for (int di = 0; di < a; ++di) {
                    for (int dj = 0; dj < b; ++dj) {
                        for (int dz = 0; dz < z; ++dz) {
                            int input_i = i * stride + di - pad;
                            int input_j = j * stride + dj - pad;

                            if (input_i >= 0 && input_i < x && input_j >= 0 && input_j < y) {
                                float input_val = data_.get()[input_i * y * z + input_j * z + dz];
                                float kernel_val = kernel.data()[di * b * z * k + dj * z * k + dz * k + filter];
                                sum += input_val * kernel_val;
                            }
                        }
                    }
                }

                output.data()[i * out_y * k + j * k + filter] = sum;
            }
        }
    }

    return output;
}

Tensor Tensor::conv3d_cpu(const Tensor& kernel, int stride, bool padding) const {
    // Input dimensions
    int batch_size = shape_[0];
    int in_channels = shape_[1];
    int depth = shape_[2];
    int height = shape_[3];
    int width = shape_[4];

    // Kernel dimensions
    int kernel_depth = kernel.shape()[0];
    int kernel_height = kernel.shape()[1];
    int kernel_width = kernel.shape()[2];
    int out_channels = kernel.shape()[4];

    // Padding
    int pad_depth = padding ? (kernel_depth - 1) / 2 : 0;
    int pad_height = padding ? (kernel_height - 1) / 2 : 0;
    int pad_width = padding ? (kernel_width - 1) / 2 : 0;

    // Output dimensions
    int output_depth = (depth - kernel_depth + 2 * pad_depth) / stride + 1;
    int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
    int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

    // Create output tensor
    Tensor output({batch_size, out_channels, output_depth, output_height, output_width}, use_gpu_);

    // Perform convolution
    for (int batch = 0; batch < batch_size; ++batch) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int od = 0; od < output_depth; ++od) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
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
                                            float input_val = data_.get()[batch * in_channels * depth * height * width +
                                                                        ic * depth * height * width +
                                                                        input_d * height * width +
                                                                        input_h * width +
                                                                        input_w];
                                            float kernel_val = kernel.data()[kd * kernel_height * kernel_width * in_channels * out_channels +
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

                        output.data()[batch * out_channels * output_depth * output_height * output_width +
                                    oc * output_depth * output_height * output_width +
                                    od * output_height * output_width +
                                    oh * output_width +
                                    ow] = sum;
                    }
                }
            }
        }
    }

    return output;
}

Tensor Tensor::conv3d(const Tensor& kernel, int stride, bool padding) const {
    if (use_gpu_) {
#ifdef USE_CUDA
        int batch_size = shape_[0];
        int in_channels = shape_[1];
        int depth = shape_[2];
        int height = shape_[3];
        int width = shape_[4];

        int kernel_depth = kernel.shape()[0];
        int kernel_height = kernel.shape()[1];
        int kernel_width = kernel.shape()[2];
        int out_channels = kernel.shape()[4];

        int pad_depth = padding ? (kernel_depth - 1) / 2 : 0;
        int pad_height = padding ? (kernel_height - 1) / 2 : 0;
        int pad_width = padding ? (kernel_width - 1) / 2 : 0;

        int output_depth = (depth - kernel_depth + 2 * pad_depth) / stride + 1;
        int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
        int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

        Tensor output({batch_size, out_channels, output_depth, output_height, output_width}, use_gpu_);

        launch_cuda_conv3d(data_.get(), kernel.data(), output.data(), batch_size, in_channels, depth, height, width,
                          kernel_depth, kernel_height, kernel_width, out_channels, stride, pad_depth, pad_height, pad_width);

        return output;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // Call the CPU implementation
        return conv3d_cpu(kernel, stride, padding);
    }
}

Tensor Tensor::conv2d(const Tensor& kernel, int stride, bool padding) const {
    if (use_gpu_) {
#ifdef USE_CUDA
        int x = shape_[0];
        int y = shape_[1];
        int z = shape_[2];
        int a = kernel.shape()[0];
        int b = kernel.shape()[1];
        int k = kernel.shape()[3];
        int pad = padding ? (a - 1) / 2 : 0;

        int out_x = (x - a + 2 * pad) / stride + 1;
        int out_y = (y - b + 2 * pad) / stride + 1;

        Tensor output({out_x, out_y, k}, use_gpu_);

        launch_cuda_conv2d(data_.get(), kernel.data(), output.data(), x, y, z, a, b, k, stride, pad);

        return output;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // Call the CPU implementation
        return conv2d_cpu(kernel, stride, padding);
    }
}

Tensor Tensor::power_cpu(float exponent) const {
    // Create output tensor with the same shape
    Tensor result(shape_, use_gpu_);

    // Perform element-wise power operation
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        result.data()[i] = std::pow(data_.get()[i], exponent);
    }

    return result;
}

Tensor Tensor::power(float exponent) const {
    if (use_gpu_) {
#ifdef USE_CUDA
        Tensor result(shape_, use_gpu_);
        size_t size = 1;
        for (int dim : shape_) size *= dim;

        launch_cuda_power(data_.get(), result.data(), exponent, size);
        return result;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // Call the CPU implementation
        return power_cpu(exponent);
    }
}

Tensor Tensor::subtract(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch in subtraction");
    }

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_subtract(data_.get(), other.data_.get(), result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            result.data()[i] = data_.get()[i] - other.data_.get()[i];
        }
    }
    return result;
}

Tensor Tensor::add_scaled(const Tensor& other, float alpha) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_add_scaled(data_.get(), other.data_.get(), alpha, result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        float* a = data_.get();
        std::vector<float>A(a, a + size);
        float* b = other.data_.get();
        std::vector<float>B(b, b + size);
        float c[size];
        std::vector<float>C(c, c + size);
        std::transform(A.begin(), A.end(), B.begin(), C.begin(),
                   [alpha](float a_elem, float b_elem) {
                       return a_elem + alpha * b_elem;
                   });
        result.load_data(C);
    }

    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_multiply(data_.get(), other.data_.get(), result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            result.data()[i] = data_.get()[i] * other.data_.get()[i];
        }
    }

    return result;
}

Tensor Tensor::divide(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_divide(data_.get(), other.data_.get(), result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            if (other.data_.get()[i] == 0) throw std::runtime_error("Division by zero");
                result.data()[i] = data_.get()[i] / other.data_.get()[i];
        }
    }

    return result;
}

Tensor Tensor::multiply_scalar(float scalar) const {
    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_multiply_scalar(data_.get(), scalar, result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            result.data()[i] = data_.get()[i] * scalar;
        }
    }

    return result;
}

Tensor Tensor::sum(int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis");
    }

    // Calculate output shape
    std::vector<int> output_shape = shape_;
    output_shape.erase(output_shape.begin() + axis);

    Tensor result(output_shape, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    // Calculate strides
    size_t stride = 1;
    for (int i = axis + 1; i < shape_.size(); ++i) {
        stride *= shape_[i];
    }

    if (use_gpu_) {
#ifdef USE_CUDA
        // Initialize result data to zero
        cudaMemset(result.data_.get(), 0, result.size() * sizeof(float));
        launch_cuda_sum(data_.get(), result.data_.get(), axis, stride, shape_[axis], size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // Initialize result data to zero
        std::fill(result.data_.get(), result.data_.get() + result.size(), 0.0f);
        // Perform sum along the axis
        for (size_t i = 0; i < size; ++i) {
            size_t output_idx = (i / (stride * shape_[axis])) * stride + (i % stride);
            result.data()[output_idx] += data_.get()[i];
        }
    }

    return result;
}

Tensor Tensor::mean(int axis) const {
    Tensor sum_result = sum(axis);
    size_t axis_size = shape_[axis];

    size_t size = 1;
    for (int dim : sum_result.shape()) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        // Correct data pointer and parameter order
        launch_cuda_mean(
            sum_result.data_.get(), 
            size, 
            static_cast<float>(axis_size) // Ensure float division
        );
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            sum_result.data()[i] /= axis_size;
        }
    }
    
    return sum_result;
}

float Tensor::max() const {
    if (use_gpu_) {
#ifdef USE_CUDA
        float result;
        launch_cuda_max(data_.get(), &result, this->size());
        return result;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        return *std::max_element(data_.get(), data_.get() + this->size());
    }
}

Tensor Tensor::max(int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis");
    }

    std::vector<int> output_shape = shape_;
    output_shape.erase(output_shape.begin() + axis);
    Tensor result(output_shape, use_gpu_);

    // Calculate dimensions
    size_t outer_dim = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= shape_[i];
    size_t axis_size = shape_[axis];
    size_t inner_stride = 1;
    for (int i = axis + 1; i < shape_.size(); ++i) inner_stride *= shape_[i];

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_max_axis(data_.get(), result.data(), outer_dim, axis_size, inner_stride);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU-based max along the specified axis
        std::fill(result.data(), result.data() + result.size(), -std::numeric_limits<float>::max());
        size_t size = this->size();
        size_t stride = 1;
        for (int i = axis + 1; i < shape_.size(); ++i) {
            stride *= shape_[i];
        }

        for (size_t i = 0; i < size; ++i) {
            size_t output_idx = (i / (stride * shape_[axis])) * stride + (i % stride);
            result.data()[output_idx] = std::max(result.data()[output_idx], data_.get()[i]);
        }
    }
    return result;
}

float Tensor::min() const {
    if (use_gpu_) {
#ifdef USE_CUDA
        float result;
        launch_cuda_min(data_.get(), &result, this->size());
        return result;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        return *std::min_element(data_.get(), data_.get() + this->size());
    }
}

// Finds the minimum value along a specified axis
Tensor Tensor::min(int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis");
    }

    std::vector<int> output_shape = shape_;
    output_shape.erase(output_shape.begin() + axis);
    Tensor result(output_shape, use_gpu_);

    size_t outer_dim = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= shape_[i];
    
    size_t axis_size = shape_[axis];
    size_t inner_stride = 1;
    for (int i = axis + 1; i < shape_.size(); ++i) 
        inner_stride *= shape_[i];

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_min_axis(
            data_.get(),
            result.data(),
            outer_dim,
            axis_size,
            inner_stride
        );
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        const float* input = data_.get();
        float* output = result.data_.get();
            
        for (size_t outer = 0; outer < outer_dim; ++outer) {
            for (size_t inner = 0; inner < inner_stride; ++inner) {
                float min_val = FLT_MAX;
                    
                for (size_t a = 0; a < axis_size; ++a) {
                    size_t input_idx = outer * axis_size * inner_stride 
                                     + a * inner_stride 
                                     + inner;
                    min_val = std::min(min_val, input[input_idx]);
                }
                    
                size_t output_idx = outer * inner_stride + inner;
                output[output_idx] = min_val;
            }
        }
    }

    return result;
}

Tensor Tensor::argmax(int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis");
    }

    std::vector<int> output_shape = shape_;
    output_shape.erase(output_shape.begin() + axis);
    Tensor result(output_shape, use_gpu_);

    // Calculate dimensions
    size_t outer_dim = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= shape_[i];
    
    size_t axis_size = shape_[axis];
    
    size_t inner_stride = 1;
    for (int i = axis + 1; i < shape_.size(); ++i) 
        inner_stride *= shape_[i];

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_argmax(
            data_.get(), 
            result.data_.get(),
            shape_.data(), 
            shape_.size(),
            axis,
            outer_dim,
            axis_size,
            inner_stride
        );
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        const float* input = data_.get();
        float* output = result.data_.get();
        
        for (size_t outer = 0; outer < outer_dim; ++outer) {
            for (size_t inner = 0; inner < inner_stride; ++inner) {
                float max_val = -FLT_MAX;
                int max_index = 0;
                
                for (size_t a = 0; a < axis_size; ++a) {
                    size_t input_idx = outer * axis_size * inner_stride 
                                     + a * inner_stride 
                                     + inner;
                    float val = input[input_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_index = a;
                    }
                }
                
                size_t output_idx = outer * inner_stride + inner;
                output[output_idx] = static_cast<float>(max_index);
            }
        }
    }

    return result;
}

// In Tensor.cpp
Tensor Tensor::argmin(int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis");
    }

    std::vector<int> output_shape = shape_;
    output_shape.erase(output_shape.begin() + axis);
    Tensor result(output_shape, use_gpu_);

    size_t outer_dim = 1;
    for (int i = 0; i < axis; ++i) outer_dim *= shape_[i];
    
    size_t axis_size = shape_[axis];
    
    size_t inner_stride = 1;
    for (int i = axis + 1; i < shape_.size(); ++i) 
        inner_stride *= shape_[i];

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_argmin(
            data_.get(), 
            result.data_.get(),
            shape_.data(), 
            shape_.size(),
            axis,
            outer_dim,
            axis_size,
            inner_stride
        );
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        const float* input = data_.get();
        float* output = result.data_.get();
        
        for (size_t outer = 0; outer < outer_dim; ++outer) {
            for (size_t inner = 0; inner < inner_stride; ++inner) {
                float min_val = FLT_MAX;
                int min_index = 0;
                
                for (size_t a = 0; a < axis_size; ++a) {
                    size_t input_idx = outer * axis_size * inner_stride 
                                     + a * inner_stride 
                                     + inner;
                    float val = input[input_idx];
                    if (val < min_val) {
                        min_val = val;
                        min_index = a;
                    }
                }
                
                size_t output_idx = outer * inner_stride + inner;
                output[output_idx] = static_cast<float>(min_index);
            }
        }
    }

    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape().size() != 2 || shape_[1] != other.shape()[0]) {
        throw std::runtime_error("Invalid shapes for matrix multiplication");
    }

    int m = shape_[0];
    int n = shape_[1];
    int p = other.shape()[1];

    Tensor result({m, p}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_matmul(data_.get(), other.data_.get(), result.data_.get(), m, n, p);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < p; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += data_.get()[i * n + k] * other.data()[k * p + j];
                }
                result.data()[i * p + j] = sum;
            }
        }
    }

    return result;
}

Tensor Tensor::einsum(const EinsumOperation& operation, const Tensor& other) const {
    // Perform the operation
    return operation(*this, other);
}

Tensor Tensor::inv() const {
    if (shape_.size() != 2 || shape_[0] != shape_[1]) {
        throw std::runtime_error("Matrix must be square to compute inverse");
    }

    const int n = shape_[0];
    Tensor result({n, n}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        float *d_A = nullptr;
        float *d_invA = nullptr;

        try {
            // Allocate device memory
            cudaMalloc(&d_A, n * n * sizeof(float));
            cudaMalloc(&d_invA, n * n * sizeof(float));

            // Copy matrix to device
            cudaMemcpy(d_A, data_.get(), n * n * sizeof(float), cudaMemcpyHostToDevice);

            // Perform inversion using cuSOLVER
            launch_cuda_inv(d_A, d_invA, n);

            // Copy result back to host
            cudaMemcpy(result.data_.get(), d_invA, n * n * sizeof(float), cudaMemcpyDeviceToHost);

            // Free device memory
            cudaFree(d_A);
            cudaFree(d_invA);
        } catch (const std::exception& e) {
            if (d_A) cudaFree(d_A);
            if (d_invA) cudaFree(d_invA);
            throw std::runtime_error("Matrix inversion failed: " + std::string(e.what()));
        }
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        Tensor augmented({n, 2 * n}, false);

        // Initialize augmented matrix [A | I]
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented.data()[i * 2 * n + j] = data_.get()[i * n + j];
                augmented.data()[i * 2 * n + j + n] = (i == j) ? 1.0f : 0.0f;
            }
        }

        // Perform Gaussian elimination
        for (int k = 0; k < n; ++k) {
            // Find pivot row
            int pivot = k;
            for (int i = k + 1; i < n; ++i) {
                if (std::abs(augmented.data()[i * 2 * n + k]) > 
                    std::abs(augmented.data()[pivot * 2 * n + k])) {
                    pivot = i;
                }
            }

            if (augmented.data()[pivot * 2 * n + k] == 0.0f) {
                throw std::runtime_error("Matrix is singular and cannot be inverted");
            }

            // Swap rows
            if (pivot != k) {
                for (int j = 0; j < 2 * n; ++j) {
                    std::swap(augmented.data()[k * 2 * n + j],
                            augmented.data()[pivot * 2 * n + j]);
                }
            }

            // Normalize pivot row
            float pivot_val = augmented.data()[k * 2 * n + k];
            for (int j = 0; j < 2 * n; ++j) {
                augmented.data()[k * 2 * n + j] /= pivot_val;
            }

            // Eliminate other rows
            for (int i = 0; i < n; ++i) {
                if (i != k) {
                    float factor = augmented.data()[i * 2 * n + k];
                    for (int j = 0; j < 2 * n; ++j) {
                        augmented.data()[i * 2 * n + j] -= 
                            factor * augmented.data()[k * 2 * n + j];
                    }
                }
            }
        }

        // Extract inverse from augmented matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result.data()[i * n + j] = augmented.data()[i * 2 * n + j + n];
            }
        }
    }

    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Transpose is only defined for 2D tensors");
    }

    int m = shape_[0];  // Original rows
    int n = shape_[1];  // Original cols
    Tensor result({n, m}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_transpose(data_.get(), result.data(), m, n);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU Implementation
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                result.data()[j * m + i] = data_.get()[i * n + j];
            }
        }
    }

    return result;
}


float Tensor::det() const {
    if (shape_.size() != 2 || shape_[0] != shape_[1]) {
        throw std::runtime_error("Matrix must be square to compute determinant");
    }

    int n = shape_[0];

    if (use_gpu_) {
#ifdef USE_CUDA
        float result;
        float* d_A;
        float* d_result;

        cudaMalloc(&d_A, n * n * sizeof(float));
        cudaMalloc(&d_result, sizeof(float));

        cudaMemcpy(d_A, data_.get(), n * n * sizeof(float), cudaMemcpyHostToDevice);

        launch_cuda_det(d_A, d_result, n);

        cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_result);

        return result;
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU Implementation (Laplace Expansion)
        if (n == 1) {
            return data_.get()[0];
        }

        float determinant = 0.0f;
        for (int j = 0; j < n; ++j) {
            Tensor submatrix({n - 1, n - 1}, use_gpu_);
            for (int i = 1; i < n; ++i) {
                for (int k = 0, l = 0; k < n; ++k) {
                    if (k == j) continue;
                    submatrix.data()[(i - 1) * (n - 1) + l] = data_.get()[i * n + k];
                    ++l;
                }
            }
            float sub_det = submatrix.det();
            determinant += (j % 2 == 0 ? 1 : -1) * data_.get()[j] * sub_det;
        }
        return determinant;
    }
}

std::pair<float, Tensor> Tensor::eig() const {
    if (shape_.size() != 2 || shape_[0] != shape_[1]) {
        throw std::runtime_error("Matrix must be square to compute eigenvalues");
    }

    const int n = shape_[0];
    Tensor eigenvector({n, 1}, use_gpu_);

    // Initialize with ones
    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_fill(eigenvector.data(), 1.0f, n);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        std::fill(eigenvector.data(), eigenvector.data() + n, 1.0f);
    }

    float eigenvalue = 0.0f;

    for (int iter = 0; iter < 100; ++iter) {
        // Calculate Rayleigh quotient
        eigenvalue = this->rayleigh_quotient(eigenvector);

        // Calculate Av and normalize
        Tensor Av = this->matmul(eigenvector);
        Tensor Av_flat = Av.flatten();
        
        float norm;
        if (use_gpu_) {
#ifdef USE_CUDA
            // Calculate norm on GPU
            float* d_norm;
            cudaMalloc(&d_norm, sizeof(float));
            cudaMemset(d_norm, 0, sizeof(float));
            
            // Custom kernel to calculate L2 norm
            launch_cuda_norm(Av_flat.data(), d_norm, n);
            cudaMemcpy(&norm, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_norm);
#else
            throw std::runtime_error("CUDA not available");
#endif
        } else {
            norm = std::sqrt(Av_flat.dot(Av_flat));
        }

        // Normalize and update eigenvector
        if (use_gpu_) {
#ifdef USE_CUDA
            launch_cuda_multiply_scalar(Av.data(), 1.0f/norm, eigenvector.data(), n);
#else
            throw std::runtime_error("CUDA not available");
#endif
        } else {
            eigenvector = Av.multiply_scalar(1.0f / norm);
        }
    }

    return {eigenvalue, eigenvector};
}

std::tuple<Tensor, Tensor, Tensor> Tensor::svd() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("SVD is only defined for 2D tensors");
    }

    int m = shape_[0];
    int n = shape_[1];
    Tensor U({m, m}, use_gpu_);
    Tensor S({std::min(m, n)}, use_gpu_);
    Tensor VT({n, n}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        // Allocate GPU memory
        float *d_A, *d_U, *d_S, *d_VT;
        cudaMalloc(&d_A, m * n * sizeof(float));
        cudaMalloc(&d_U, m * m * sizeof(float));
        cudaMalloc(&d_S, std::min(m, n) * sizeof(float));
        cudaMalloc(&d_VT, n * n * sizeof(float));

        // Copy data to GPU
        cudaMemcpy(d_A, data_.get(), m * n * sizeof(float), cudaMemcpyHostToDevice);

        // Launch CUDA SVD
        launch_cuda_svd(d_A, d_U, d_S, d_VT, m, n);

        // Copy results back to CPU
        cudaMemcpy(U.data(), d_U, m * m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(S.data(), d_S, std::min(m, n) * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(VT.data(), d_VT, n * n * sizeof(float), cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(d_A);
        cudaFree(d_U);
        cudaFree(d_S);
        cudaFree(d_VT);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // Use Eigen for CPU SVD
        Eigen::MatrixXf A = Eigen::Map<Eigen::MatrixXf>(data_.get(), m, n);
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Copy data to Tensor objects
        Eigen::MatrixXf U_mat = svd.matrixU();
        Eigen::VectorXf S_vec = svd.singularValues();
        Eigen::MatrixXf VT_mat = svd.matrixV().transpose();

        std::memcpy(U.data(), U_mat.data(), m * m * sizeof(float));
        std::memcpy(S.data(), S_vec.data(), std::min(m, n) * sizeof(float));
        std::memcpy(VT.data(), VT_mat.data(), n * n * sizeof(float));
    }

    return {U, S, VT};
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_size = 1;
    for (int dim : new_shape) new_size *= dim;

    size_t current_size = 1;
    for (int dim : shape_) current_size *= dim;

    if (new_size != current_size) {
        throw std::runtime_error("Total size of new shape must match the original size");
    }

    Tensor result(new_shape, use_gpu_);
    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_reshape(data_.get(), result.data_.get(), new_size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU reshape logic: just copy the data (reshape doesn't change the data itself)
        std::copy(data_.get(), data_.get() + current_size, result.data_.get());
    }

    return result;
}

Tensor Tensor::flatten() const {
    std::vector<int> new_shape{static_cast<int>(this->size())};
    
    Tensor result(new_shape, use_gpu_);
    size_t bytes = size() * sizeof(float);

    if (use_gpu_) {
#ifdef USE_CUDA
        cudaMemcpy(result.data(), data(), bytes, cudaMemcpyDeviceToDevice);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        std::copy(data(), data() + size(), result.data());
    }
    
    return result;
}

Tensor Tensor::expand_dims(int axis) const {
    if (axis < 0 || axis > shape_.size()) {
        throw std::runtime_error("Invalid axis for expand_dims");
    }

    std::vector<int> new_shape = shape_;
    new_shape.insert(new_shape.begin() + axis, 1);

    Tensor result(new_shape, use_gpu_);
    size_t new_size = 1;
    for (int dim : new_shape) new_size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_expand_dims(data_.get(), result.data_.get(), new_size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU expand_dims: reshape logic with a new dimension
        std::copy(data_.get(), data_.get() + new_size, result.data_.get());
    }

    return result;
}

Tensor Tensor::squeeze() const {
    std::vector<size_t> new_shape;  // Use size_t instead of int
    for (int dim : shape_) {
        if (dim != 1) {
            new_shape.push_back(dim);
        }
    }

    if (new_shape.empty()) {
        new_shape.push_back(1); // Ensure at least 1 dimension
    }

    // Convert new_shape to std::vector<int> before passing it to the constructor
    std::vector<int> int_new_shape(new_shape.begin(), new_shape.end());
    Tensor result(int_new_shape, use_gpu_);
    size_t total_size = 1;
    for (size_t dim : new_shape) total_size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        // Cast const int* to const size_t* to avoid the "const qualifier" issue
        launch_cuda_squeeze(data_.get(), result.data_.get(), total_size, new_shape.data(), const_cast<size_t*>(reinterpret_cast<const size_t*>(shape_.data())));
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU squeeze: just copy the data while skipping size 1 dimensions
        std::copy(data_.get(), data_.get() + total_size, result.data_.get());
    }

    return result;
}

Tensor Tensor::concat(const Tensor& other, int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis for concat");
    }

    // Check if shapes are compatible for concatenation
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i != axis && shape_[i] != other.shape_[i]) {
            throw std::runtime_error("Tensors must have compatible shapes for concatenation");
        }
    }

    // Calculate the new shape
    std::vector<int> new_shape = shape_;
    new_shape[axis] += other.shape_[axis]; // Concatenate along the given axis

    Tensor result(new_shape, use_gpu_);
    size_t total_size = 1;
    for (int dim : new_shape) total_size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        size_t size1 = 1;
        for (int dim : shape_) size1 *= dim;
        size_t size2 = 1;
        for (int dim : other.shape_) size2 *= dim;

        launch_cuda_concat(data_.get(), other.data_.get(), result.data_.get(), size1, size2, axis, shape_[axis], other.shape_[axis]);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU concat: manually concatenate the two tensors along the axis
        size_t size1 = 1;
        for (int dim : shape_) size1 *= dim;
        size_t size2 = 1;
        for (int dim : other.shape_) size2 *= dim;

        // Copy data from first tensor
        std::copy(data_.get(), data_.get() + size1, result.data_.get());

        // Copy data from second tensor, adjusted for concatenation axis
        std::copy(other.data_.get(), other.data_.get() + size2, result.data_.get() + size1);
    }

    return result;
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int axis) {
    if (tensors.empty()) {
        throw std::runtime_error("No tensors provided for stacking");
    }

    // Check that all tensors have the same shape
    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].shape() != tensors[0].shape()) {
            throw std::runtime_error("All tensors must have the same shape for stacking");
        }
    }

    // Create the new shape
    std::vector<int> new_shape = tensors[0].shape();
    new_shape.insert(new_shape.begin() + axis, tensors.size());

    // Convert new_shape to size_t for CUDA compatibility
    std::vector<size_t> size_t_new_shape(new_shape.begin(), new_shape.end());

    // Create result tensor
    Tensor result(new_shape, tensors[0].use_gpu());
    
    size_t tensor_size = 1;
    for (int dim : tensors[0].shape()) tensor_size *= dim;

    if (tensors[0].use_gpu()) {
#ifdef USE_CUDA
        size_t offset = 0;
        for (const Tensor& t : tensors) {
            cudaError_t err = cudaMemcpy(
                result.data_.get() + offset,
                t.data_.get(),
                tensor_size * sizeof(float),
                cudaMemcpyDeviceToDevice
            );
            
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed: " + 
                    std::string(cudaGetErrorString(err)));
            }
            offset += tensor_size;
        }
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // For CPU, just copy the data as usual
        size_t offset = 0;
        for (const Tensor& tensor : tensors) {
            std::copy(tensor.data(), tensor.data() + tensor_size, result.data_.get() + offset);
            offset += tensor_size;
        }
    }

    return result;
}

Tensor Tensor::permute(const std::vector<int>& new_order) const {
    if (new_order.size() != shape_.size()) {
        throw std::runtime_error("New order must have the same number of dimensions as the tensor");
    }

    // Check that all the axes in new_order are valid
    std::vector<int> new_shape;
    for (int axis : new_order) {
        if (axis < 0 || axis >= shape_.size()) {
            throw std::runtime_error("Invalid axis in new order");
        }
        new_shape.push_back(shape_[axis]);
    }

    Tensor result(new_shape, use_gpu_);

    // Calculate the size of the data
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        // Copy shape and new_order to raw arrays
        std::vector<int> shape_vec = shape_;
        std::vector<int> new_order_vec = new_order;

        // Allocate device memory for shape and new_order
        int* d_shape;
        int* d_new_order;
        cudaMalloc(&d_shape, shape_vec.size() * sizeof(int));
        cudaMalloc(&d_new_order, new_order_vec.size() * sizeof(int));
        cudaMemcpy(d_shape, shape_vec.data(), shape_vec.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_new_order, new_order_vec.data(), new_order_vec.size() * sizeof(int), cudaMemcpyHostToDevice);

        // Launch CUDA kernel for permutation
        launch_cuda_permute(data_.get(), result.data_.get(), d_shape, d_new_order, shape_vec.size(), size);

        // Free device memory
        cudaFree(d_shape);
        cudaFree(d_new_order);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation with proper permutation
        const size_t ndims = shape_.size();
        std::vector<size_t> strides(ndims, 1);
        std::vector<size_t> new_strides(ndims, 1);

        // Calculate strides for original tensor
        for (int i = ndims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape_[i + 1];
        }

        // Calculate strides for new tensor
        for (int i = ndims - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        // Create inverse permutation mapping
        std::vector<size_t> inverse_perm(ndims);
        for (size_t i = 0; i < ndims; ++i) {
            inverse_perm[new_order[i]] = i;
        }

        // Perform permutation
        const float* src = data_.get();
        float* dst = result.data_.get();
        const size_t total_elements = result.size();

        for (size_t idx = 0; idx < total_elements; ++idx) {
            size_t temp = idx;
            size_t original_idx = 0;
            
            // Calculate original coordinates
            for (int i = ndims - 1; i >= 0; --i) {
                const size_t dim = inverse_perm[i];
                const size_t stride = strides[i];
                const size_t coord = temp / new_strides[dim];
                temp %= new_strides[dim];
                original_idx += coord * stride;
            }
            
            dst[idx] = src[original_idx];
        }
    }

    return result;
}


// Helper function to repeat
Tensor Tensor::repeat_gpu(int axis, int repeats) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis for repeat");
    }

    // Calculate the new shape
    std::vector<int> new_shape = shape_;
    new_shape[axis] *= repeats;

    Tensor result(new_shape, use_gpu_);

    // Calculate the size of the data
    size_t size = 1;
    for (int dim : new_shape) size *= dim;

    // Copy shapes to raw arrays
    std::vector<int> input_shape_vec = shape_;
    std::vector<int> output_shape_vec = new_shape;

    // Launch CUDA kernel for repeat
    launch_cuda_repeat(data_.get(), result.data_.get(), input_shape_vec.data(), output_shape_vec.data(), shape_.size(), axis, size);

    return result;
}

// Helper function to compute broadcasted shape
std::vector<int> Tensor::broadcast_shapes(const std::vector<int>& shape1, const std::vector<int>& shape2) {
    std::vector<int> result_shape;
    int max_dims = std::max(shape1.size(), shape2.size());

    // Pad shapes with 1s on the left to make them the same length
    std::vector<int> padded_shape1(max_dims, 1);
    std::vector<int> padded_shape2(max_dims, 1);

    std::copy(shape1.rbegin(), shape1.rend(), padded_shape1.rbegin());
    std::copy(shape2.rbegin(), shape2.rend(), padded_shape2.rbegin());

    // Compute the broadcasted shape
    for (int i = 0; i < max_dims; ++i) {
        int dim1 = padded_shape1[i];
        int dim2 = padded_shape2[i];

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::runtime_error("Shapes are not broadcastable");
        }

        result_shape.push_back(std::max(dim1, dim2));
    }

    return result_shape;
}

// Broadcast two tensors to the same shape
std::pair<Tensor, Tensor> Tensor::broadcast_tensors(const Tensor& A, const Tensor& B) {
    std::vector<int> broadcasted_shape = broadcast_shapes(A.shape(), B.shape());

    Tensor A_broadcasted = A;
    Tensor B_broadcasted = B;

    // Expand dimensions of A if necessary
    while (A_broadcasted.shape().size() < broadcasted_shape.size()) {
        A_broadcasted = A_broadcasted.expand_dims(0);
    }

    // Expand dimensions of B if necessary
    while (B_broadcasted.shape().size() < broadcasted_shape.size()) {
        B_broadcasted = B_broadcasted.expand_dims(0);
    }

    // Repeat data along new dimensions for A
    for (size_t i = 0; i < broadcasted_shape.size(); ++i) {
        if (A_broadcasted.shape()[i] == 1 && broadcasted_shape[i] > 1) {
            if (A_broadcasted.use_gpu()) {
#ifdef USE_CUDA
                A_broadcasted = A_broadcasted.repeat_gpu(i, broadcasted_shape[i]);
#else
                throw std::runtime_error("CUDA not available");
#endif
            } else {
                A_broadcasted = A_broadcasted.repeat(i, broadcasted_shape[i]);
            }
        }
    }

    // Repeat data along new dimensions for B
    for (size_t i = 0; i < broadcasted_shape.size(); ++i) {
        if (B_broadcasted.shape()[i] == 1 && broadcasted_shape[i] > 1) {
            if (B_broadcasted.use_gpu()) {
#ifdef USE_CUDA
                B_broadcasted = B_broadcasted.repeat_gpu(i, broadcasted_shape[i]);
#else
                throw std::runtime_error("CUDA not available");
#endif
            } else {
                B_broadcasted = B_broadcasted.repeat(i, broadcasted_shape[i]);
            }
        }
    }

    return {A_broadcasted, B_broadcasted};
}

// Element-wise equality comparison
Tensor Tensor::operator==(const Tensor& other) const {
    auto [A_broadcasted, B_broadcasted] = broadcast_tensors(*this, other);

    Tensor result(A_broadcasted.shape(), use_gpu_);
    size_t size = 1;
    for (int dim : A_broadcasted.shape()) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_equal(A_broadcasted.data(), B_broadcasted.data(), result.data(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            result.data()[i] = (A_broadcasted.data()[i] == B_broadcasted.data()[i]) ? 1.0f : 0.0f;
        }
    }

    return result;
}

Tensor Tensor::repeat(int axis, int repeats) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis for repeat");
    }

    if (repeats < 1) {
        throw std::runtime_error("Number of repeats must be at least 1");
    }

    // Compute the new shape
    std::vector<int> new_shape = shape_;
    new_shape[axis] *= repeats;

    Tensor result(new_shape, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        // Calculate the size of the data
        size_t size = 1;
        for (int dim : new_shape) size *= dim;

        // Copy shapes to raw arrays
        std::vector<int> input_shape_vec = shape_;
        std::vector<int> output_shape_vec = new_shape;

        // Launch CUDA kernel for repeat
        launch_cuda_repeat(data_.get(), result.data_.get(), input_shape_vec.data(), output_shape_vec.data(), shape_.size(), axis, size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation: repeat data along the specified axis
        size_t outer_size = 1;
        for (int i = 0; i < axis; ++i) {
            outer_size *= shape_[i];
        }

        size_t inner_size = 1;
        for (int i = axis + 1; i < shape_.size(); ++i) {
            inner_size *= shape_[i];
        }

        size_t axis_size = shape_[axis];

        for (size_t outer = 0; outer < outer_size; ++outer) {
            for (size_t r = 0; r < repeats; ++r) {
                for (size_t a = 0; a < axis_size; ++a) {
                    for (size_t inner = 0; inner < inner_size; ++inner) {
                        size_t input_idx = outer * axis_size * inner_size + a * inner_size + inner;
                        size_t output_idx = outer * repeats * axis_size * inner_size + r * axis_size * inner_size + a * inner_size + inner;
                        result.data()[output_idx] = data_.get()[input_idx];
                    }
                }
            }
        }
    }

    return result;
}

// Boolean indexing
Tensor Tensor::operator[](const Tensor& mask) const {
    if (mask.shape() != shape_) {
        throw std::runtime_error("Mask shape must match tensor shape");
    }

    // Calculate the size of the tensor data
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    // Count the number of true values in the mask
    size_t count = 0;
    for (size_t i = 0; i < size; ++i) {
        if (mask.data_.get()[i] != 0.0f) {
            ++count;
        }
    }

    // Create a result tensor with the selected elements
    Tensor result({static_cast<int>(count)}, use_gpu_);

    // Copy the selected elements
    size_t index = 0;
    for (size_t i = 0; i < size; ++i) {
        if (mask.data_.get()[i] != 0.0f) {
            result.data_.get()[index++] = data_.get()[i];
        }
    }

    return result;
}

Tensor& Tensor::operator=(const std::pair<Tensor, float>& masked_assignment) {
    const Tensor& mask = masked_assignment.first;
    float value = masked_assignment.second;

    if (mask.shape() != shape_) {
        throw std::runtime_error("Mask shape must match tensor shape");
    }

    // Calculate the size of the tensor data
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        // Launch CUDA kernel for masked assignment
        launch_cuda_masked_assign(data_.get(), mask.data_.get(), value, size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation: set elements where the mask is true to the specified value
        for (size_t i = 0; i < size; ++i) {
            if (mask.data_.get()[i] != 0.0f) {
                data_.get()[i] = value;
            }
        }
    }

    return *this;
}

// Helper function to create a boolean tensor from a condition
Tensor Tensor::from_condition(const Tensor& condition) {
    Tensor result(condition.shape(), condition.use_gpu());

    // Calculate the size of the tensor data
    size_t size = 1;
    for (int dim : condition.shape()) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        result.data_.get()[i] = (condition.data_.get()[i] != 0.0f) ? 1.0f : 0.0f;
    }

    return result;
}

Tensor Tensor::operator>(float scalar) const {
    Tensor result(shape_, use_gpu_);

    // Calculate the size of the tensor data
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_greater_than_scalar(data_.get(), result.data_.get(), scalar, size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        for (size_t i = 0; i < size; ++i) {
            result.data_.get()[i] = (data_.get()[i] > scalar) ? 1.0f : 0.0f;
        }
    }

    return result;
}

Tensor Tensor::operator>(const Tensor& other) const {
    // Broadcast shapes if necessary
    auto [A_broadcasted, B_broadcasted] = broadcast_tensors(*this, other);

    Tensor result(A_broadcasted.shape(), use_gpu_);

    // Calculate the size of the tensor data
    size_t size = 1;
    for (int dim : A_broadcasted.shape()) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_greater_than_tensor(A_broadcasted.data_.get(), B_broadcasted.data_.get(), result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        for (size_t i = 0; i < size; ++i) {
            result.data_.get()[i] = (A_broadcasted.data_.get()[i] > B_broadcasted.data_.get()[i]) ? 1.0f : 0.0f;
        }
    }

    return result;
}



Tensor Tensor::maxpool(int kernel_size, int stride, bool padding) const {
    if (shape_.size() != 3) {
        throw std::runtime_error("Max pooling is only supported for 3D tensors (batch_size, channels, length)");
    }

    int batch_size = shape_[0];
    int channels = shape_[1];
    int length = shape_[2];

    int pad = padding ? (kernel_size - 1) / 2 : 0;

    int output_length = (length - kernel_size + 2 * pad) / stride + 1;

    Tensor output({batch_size, channels, output_length}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_maxpool(data_.get(), output.data(), batch_size, channels, length, kernel_size, stride, pad, output_length);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int ch = 0; ch < channels; ++ch) {
                for (int ol = 0; ol < output_length; ++ol) {
                    float max_val = -std::numeric_limits<float>::max();

                    for (int ks = 0; ks < kernel_size; ++ks) {
                        int input_pos = ol * stride + ks - pad;

                        if (input_pos >= 0 && input_pos < length) {
                            float val = data_.get()[batch * channels * length + ch * length + input_pos];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }

                    output.data()[batch * channels * output_length + ch * output_length + ol] = max_val;
                }
            }
        }
    }

    return output;
}

Tensor Tensor::avgpool(int kernel_size, int stride, bool padding) const {
    if (shape_.size() != 3) {
        throw std::runtime_error("Average pooling is only supported for 3D tensors (batch_size, channels, length)");
    }

    int batch_size = shape_[0];
    int channels = shape_[1];
    int length = shape_[2];

    int pad = padding ? (kernel_size - 1) / 2 : 0;

    int output_length = (length - kernel_size + 2 * pad) / stride + 1;

    Tensor output({batch_size, channels, output_length}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_avgpool(data_.get(), output.data(), batch_size, channels, length, kernel_size, stride, pad, output_length);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int ch = 0; ch < channels; ++ch) {
                for (int ol = 0; ol < output_length; ++ol) {
                    float sum = 0.0f;
                    int count = 0;

                    for (int ks = 0; ks < kernel_size; ++ks) {
                        int input_pos = ol * stride + ks - pad;

                        if (input_pos >= 0 && input_pos < length) {
                            float val = data_.get()[batch * channels * length + ch * length + input_pos];
                            sum += val;
                            count++;
                        }
                    }

                    output.data()[batch * channels * output_length + ch * output_length + ol] = sum / count;
                }
            }
        }
    }

    return output;
}

Tensor Tensor::maxpool2d(int kernel_height, int kernel_width, int stride = 1, bool padding = false) const {
    if (shape_.size() != 4) {
        throw std::runtime_error("2D max pooling is only supported for 4D tensors (batch_size, channels, height, width)");
    }

    int batch_size = shape_[0];
    int channels = shape_[1];
    int height = shape_[2];
    int width = shape_[3];

    int pad_height = padding ? (kernel_height - 1) / 2 : 0;
    int pad_width = padding ? (kernel_width - 1) / 2 : 0;

    int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
    int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

    Tensor output({batch_size, channels, output_height, output_width}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_maxpool2d(data_.get(), output.data_.get(), batch_size, channels, height, width,
                              kernel_height, kernel_width, stride, pad_height, pad_width);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int ch = 0; ch < channels; ++ch) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        float max_val = -std::numeric_limits<float>::max();

                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int input_h = oh * stride + kh - pad_height;
                                int input_w = ow * stride + kw - pad_width;

                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    float val = data_.get()[batch * channels * height * width +
                                                            ch * height * width +
                                                            input_h * width +
                                                            input_w];
                                    if (val > max_val) {
                                        max_val = val;
                                    }
                                }
                            }
                        }

                        output.data()[batch * channels * output_height * output_width +
                                     ch * output_height * output_width +
                                     oh * output_width +
                                     ow] = max_val;
                    }
                }
            }
        }
    }

    return output;
}

Tensor Tensor::avgpool2d(int kernel_height, int kernel_width, int stride = 1, bool padding = false) const {
    if (shape_.size() != 4) {
        throw std::runtime_error("2D average pooling is only supported for 4D tensors (batch_size, channels, height, width)");
    }

    int batch_size = shape_[0];
    int channels = shape_[1];
    int height = shape_[2];
    int width = shape_[3];

    int pad_height = padding ? (kernel_height - 1) / 2 : 0;
    int pad_width = padding ? (kernel_width - 1) / 2 : 0;

    int output_height = (height - kernel_height + 2 * pad_height) / stride + 1;
    int output_width = (width - kernel_width + 2 * pad_width) / stride + 1;

    Tensor output({batch_size, channels, output_height, output_width}, use_gpu_);

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_avgpool2d(data_.get(), output.data_.get(), batch_size, channels, height, width,
                              kernel_height, kernel_width, stride, pad_height, pad_width);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        // CPU implementation
        for (int batch = 0; batch < batch_size; ++batch) {
            for (int ch = 0; ch < channels; ++ch) {
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        float sum = 0.0f;
                        int count = 0;

                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                int input_h = oh * stride + kh - pad_height;
                                int input_w = ow * stride + kw - pad_width;

                                if (input_h >= 0 && input_h < height && input_w >= 0 && input_w < width) {
                                    float val = data_.get()[batch * channels * height * width +
                                                            ch * height * width +
                                                            input_h * width +
                                                            input_w];
                                    sum += val;
                                    count++;
                                }
                            }
                        }

                        output.data()[batch * channels * output_height * output_width +
                                     ch * output_height * output_width +
                                     oh * output_width +
                                     ow] = sum / count;
                    }
                }
            }
        }
    }

    return output;
}

