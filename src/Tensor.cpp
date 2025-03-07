#include "Tensor.h"
#include "cuda_kernels.h"
#include <iostream>

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

void Tensor::allocate_memory() {
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        float* gpu_data;
        cudaMalloc(&gpu_data, size * sizeof(float));
        data_ = std::shared_ptr<float>(gpu_data, [](float* ptr) { cudaFree(ptr); });
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        data_ = std::shared_ptr<float>(new float[size], [](float* ptr) { delete[] ptr; });
    }
}

void Tensor::free_memory() {
    // Memory is automatically managed by shared_ptr
}

Tensor Tensor::add(const Tensor& other, float alpha) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    if (use_gpu_) {
#ifdef USE_CUDA
        launch_cuda_add(data_.get(), other.data_.get(), alpha, result.data_.get(), size);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        for (size_t i = 0; i < size; ++i) {
            result.data_.get()[i] = data_.get()[i] + alpha * other.data_.get()[i];
        }
    }

    return result;
}

float Tensor::dot(const Tensor& other) const {
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
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        result.data()[i] = data_.get()[i] - other.data_.get()[i];
    }

    return result;
}

Tensor Tensor::add_scaled(const Tensor& other, float alpha) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        result.data()[i] = data_.get()[i] + alpha * other.data_.get()[i];
    }

    return result;
}

Tensor Tensor::multiply(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        result.data()[i] = data_.get()[i] * other.data_.get()[i];
    }

    return result;
}

Tensor Tensor::divide(const Tensor& other) const {
    if (shape_ != other.shape_) throw std::runtime_error("Shape mismatch");

    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        if (other.data_.get()[i] == 0) throw std::runtime_error("Division by zero");
        result.data()[i] = data_.get()[i] / other.data_.get()[i];
    }

    return result;
}

Tensor Tensor::multiply_scalar(float scalar) const {
    Tensor result(shape_, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        result.data()[i] = data_.get()[i] * scalar;
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

    // Perform sum along the axis
    for (size_t i = 0; i < size; ++i) {
        size_t output_idx = (i / (stride * shape_[axis])) * stride + (i % stride);
        result.data()[output_idx] += data_.get()[i];
    }

    return result;
}

Tensor Tensor::mean(int axis) const {
    Tensor sum_result = sum(axis);
    size_t axis_size = shape_[axis];

    // Divide by the size of the axis to compute the mean
    size_t size = 1;
    for (int dim : sum_result.shape()) size *= dim;

    for (size_t i = 0; i < size; ++i) {
        sum_result.data()[i] /= axis_size;
    }

    return sum_result;
}

Tensor Tensor::max(int axis) const {
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

    // Initialize result with the smallest possible float value
    std::fill(result.data(), result.data() + result.shape()[0] * result.shape()[1], -std::numeric_limits<float>::max());

    // Perform max along the axis
    for (size_t i = 0; i < size; ++i) {
        size_t output_idx = (i / (stride * shape_[axis])) * stride + (i % stride);
        result.data()[output_idx] = std::max(result.data()[output_idx], data_.get()[i]);
    }

    return result;
}

Tensor Tensor::min(int axis) const {
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

    // Initialize result with the largest possible float value
    std::fill(result.data(), result.data() + result.shape()[0] * result.shape()[1], std::numeric_limits<float>::max());

    // Perform min along the axis
    for (size_t i = 0; i < size; ++i) {
        size_t output_idx = (i / (stride * shape_[axis])) * stride + (i % stride);
        result.data()[output_idx] = std::min(result.data()[output_idx], data_.get()[i]);
    }

    return result;
}

Tensor Tensor::argmax(int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis");
    }

    // Calculate output shape
    std::vector<int> output_shape = shape_;
    output_shape.erase(output_shape.begin() + axis);

    // Create a tensor to store indices (integers)
    Tensor result(output_shape, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    // Calculate strides
    size_t stride = 1;
    for (int i = axis + 1; i < shape_.size(); ++i) {
        stride *= shape_[i];
    }

    // Initialize result with zeros
    std::fill(result.data(), result.data() + result.shape()[0] * result.shape()[1], 0);

    // Perform argmax along the axis
    for (size_t i = 0; i < size; ++i) {
        size_t output_idx = (i / (stride * shape_[axis])) * stride + (i % stride);
        size_t current_idx = (i / stride) % shape_[axis];

        if (data_.get()[i] > data_.get()[output_idx * shape_[axis] + static_cast<int>(result.data()[output_idx])]) {
            result.data()[output_idx] = static_cast<float>(current_idx);
        }
    }

    return result;
}

Tensor Tensor::argmin(int axis) const {
    if (axis < 0 || axis >= shape_.size()) {
        throw std::runtime_error("Invalid axis");
    }

    // Calculate output shape
    std::vector<int> output_shape = shape_;
    output_shape.erase(output_shape.begin() + axis);

    // Create a tensor to store indices (integers)
    Tensor result(output_shape, use_gpu_);
    size_t size = 1;
    for (int dim : shape_) size *= dim;

    // Calculate strides
    size_t stride = 1;
    for (int i = axis + 1; i < shape_.size(); ++i) {
        stride *= shape_[i];
    }

    // Initialize result with zeros
    std::fill(result.data(), result.data() + result.shape()[0] * result.shape()[1], 0);

    // Perform argmin along the axis
    for (size_t i = 0; i < size; ++i) {
        size_t output_idx = (i / (stride * shape_[axis])) * stride + (i % stride);
        size_t current_idx = (i / stride) % shape_[axis];

        if (data_.get()[i] < data_.get()[output_idx * shape_[axis] + static_cast<int>(result.data()[output_idx])]) {
            result.data()[output_idx] = static_cast<float>(current_idx);
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

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += data_.get()[i * n + k] * other.data()[k * p + j];
            }
            result.data()[i * p + j] = sum;
        }
    }

    return result;
}

Tensor Tensor::einsum(const std::string& equation, const Tensor& other) const {
    std::cout << "Processing einsum equation: " << equation << std::endl; // Debugging

    // Split the equation into input and output parts
    size_t arrow_pos = equation.find("->");
    std::string input_eq = equation.substr(0, arrow_pos);
    std::string output_eq = (arrow_pos == std::string::npos) ? "" : equation.substr(arrow_pos + 2);

    // Split input_eq into individual tensor indices
    size_t comma_pos = input_eq.find(',');
    std::string indices_A = input_eq.substr(0, comma_pos);
    std::string indices_B = input_eq.substr(comma_pos + 1);

    // Determine the output shape
    std::vector<int> output_shape;
    for (char c : output_eq) {
        int dim_A = (indices_A.find(c) != std::string::npos) ? shape_[indices_A.find(c)] : -1;
        int dim_B = (indices_B.find(c) != std::string::npos) ? other.shape()[indices_B.find(c)] : -1;
        if (dim_A != -1 && dim_B != -1 && dim_A != dim_B) {
            throw std::runtime_error("Dimension mismatch in einsum");
        }
        output_shape.push_back((dim_A != -1) ? dim_A : dim_B);
    }

    // Create the output tensor
    Tensor result(output_shape, use_gpu_);

    // Perform the summation
    if (equation == "ij,jk->ik") {
        // Matrix multiplication
        int m = shape_[0];
        int n = shape_[1];
        int p = other.shape()[1];

        for (int i = 0; i < m; ++i) {
            for (int k = 0; k < p; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    sum += data_.get()[i * n + j] * other.data()[j * p + k];
                }
                result.data()[i * p + k] = sum;
            }
        }
    } else if (equation == "i,i->") {
        // Dot product
        int n = shape_[0];
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += data_.get()[i] * other.data()[i];
        }
        result.data()[0] = sum;
    } else {
        throw std::runtime_error("Unsupported einsum equation: " + equation);
    }

    return result;
}

Tensor Tensor::inv() const {
    if (shape_.size() != 2 || shape_[0] != shape_[1]) {
        throw std::runtime_error("Matrix must be square to compute inverse");
    }

    int n = shape_[0];
    Tensor result({n, n}, use_gpu_);
    Tensor augmented({n, 2 * n}, use_gpu_);

    // Initialize augmented matrix [A | I]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented.data()[i * 2 * n + j] = data_.get()[i * n + j];
            augmented.data()[i * 2 * n + j + n] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Perform Gaussian elimination
    for (int i = 0; i < n; ++i) {
        // Find the pivot
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(augmented.data()[j * 2 * n + i]) > std::abs(augmented.data()[pivot * 2 * n + i])) {
                pivot = j;
            }
        }

        // Swap rows
        if (pivot != i) {
            for (int j = 0; j < 2 * n; ++j) {
                std::swap(augmented.data()[i * 2 * n + j], augmented.data()[pivot * 2 * n + j]);
            }
        }

        // Normalize the pivot row
        float pivot_value = augmented.data()[i * 2 * n + i];
        if (pivot_value == 0.0f) {
            throw std::runtime_error("Matrix is singular and cannot be inverted");
        }

        for (int j = 0; j < 2 * n; ++j) {
            augmented.data()[i * 2 * n + j] /= pivot_value;
        }

        // Eliminate other rows
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                float factor = augmented.data()[j * 2 * n + i];
                for (int k = 0; k < 2 * n; ++k) {
                    augmented.data()[j * 2 * n + k] -= factor * augmented.data()[i * 2 * n + k];
                }
            }
        }
    }

    // Extract the inverse from the augmented matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result.data()[i * n + j] = augmented.data()[i * 2 * n + j + n];
        }
    }

    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::runtime_error("Transpose is only defined for 2D tensors");
    }

    int m = shape_[0];
    int n = shape_[1];
    Tensor result({n, m}, use_gpu_);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result.data()[j * m + i] = data_.get()[i * n + j];
        }
    }

    return result;
}

float Tensor::det() const {
    if (shape_.size() != 2 || shape_[0] != shape_[1]) {
        throw std::runtime_error("Matrix must be square to compute determinant");
    }

    int n = shape_[0];
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

std::pair<float, Tensor> Tensor::eig() const {
    if (shape_.size() != 2 || shape_[0] != shape_[1]) {
        throw std::runtime_error("Matrix must be square to compute eigenvalues");
    }

    int n = shape_[0];
    Tensor eigenvector({n, 1}, use_gpu_);
    std::fill(eigenvector.data(), eigenvector.data() + n, 1.0f);

    float eigenvalue = 0.0f;
    for (int iter = 0; iter < 100; ++iter) {
        Tensor new_eigenvector = matmul(eigenvector);
        float norm = 0.0f;
        for (int i = 0; i < n; ++i) {
            norm += new_eigenvector.data()[i] * new_eigenvector.data()[i];
        }
        norm = std::sqrt(norm);

        for (int i = 0; i < n; ++i) {
            eigenvector.data()[i] = new_eigenvector.data()[i] / norm;
        }

        eigenvalue = norm;
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
    Tensor S({m, n}, use_gpu_);
    Tensor V({n, n}, use_gpu_);

    // Placeholder for actual SVD computation
    throw std::runtime_error("SVD not implemented");
}

