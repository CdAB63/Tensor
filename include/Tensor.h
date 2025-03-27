#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <functional>
#include "cuda_kernels.h"
#include <iostream>
#include <cuda_runtime.h>
#include <Eigen/Dense>

class Tensor {
public:
    using EinsumOperation = std::function<Tensor(const Tensor&, const Tensor&)>;

    Tensor(const std::vector<int>& shape, bool use_gpu = false);
    ~Tensor();

    // Memory management
    void load_data(const std::vector<float>& data);
    std::vector<float> get_data() const;

    // Utility method to calculate the total number of elements in the tensor
    size_t size() const {
        size_t size = 1;
        for (int dim : shape_) size *= dim;
        return size;
    }
    
    // Basic operations
    bool use_gpu() const;
    Tensor copy() const;
    float rayleigh_quotient(const Tensor& v) const;
    Tensor add(const Tensor& other) const;
    float dot(const Tensor& other) const;
    Tensor conv1d(const Tensor& kernel, int stride, bool padding) const;
    Tensor conv1d_cpu(const Tensor& kernel, int stride, bool padding) const;
    Tensor conv2d(const Tensor& kernel, int stride, bool padding) const;
    Tensor conv2d_cpu(const Tensor& kernel, int stride, bool padding) const;
    Tensor conv3d(const Tensor& kernel, int stride, bool padding) const;
    Tensor conv3d_cpu(const Tensor& kernel, int stride, bool padding) const;
    Tensor power(float exponent) const;
    Tensor subtract(const Tensor& other) const;
    Tensor add_scaled(const Tensor& other, float alpha) const;
    Tensor multiply(const Tensor& other) const;
    Tensor divide(const Tensor& other) const;
    Tensor multiply_scalar(float scalar) const;
    Tensor sum(int axis) const;
    Tensor mean(int axis) const;
    float max() const; // The max
    Tensor max(int axis) const;
    float min() const;
    Tensor min(int axis) const;
    Tensor argmax(int axis) const;
    Tensor argmin(int axis) const;
    Tensor matmul(const Tensor& other) const; // Matrix multiplication
    Tensor inv() const; // Inverse of a square matrix
    Tensor transpose() const; // Transpose of a matrix
    float det() const; // Determinant of a square matrix
    std::pair<float, Tensor> eig() const; // Eigenvalues and eigenvectors
    std::tuple<Tensor, Tensor, Tensor> svd() const; // Singular Value Decomposition
    Tensor einsum(const EinsumOperation& operation, const Tensor& other) const;
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor flatten() const;
    Tensor expand_dims(int axis) const;
    Tensor squeeze() const;
    Tensor concat(const Tensor& other, int axis) const;
    static Tensor stack(const std::vector<Tensor>& tensors, int axis);
    Tensor permute(const std::vector<int>& new_order) const;
    Tensor repeat_gpu(int axis, int repeats) const;
    static std::vector<int> broadcast_shapes(const std::vector<int>& shape1, const std::vector<int>& shape2);
    static std::pair<Tensor, Tensor> broadcast_tensors(const Tensor& A, const Tensor& B);
    Tensor repeat(int axis, int repeats) const;
    Tensor operator[](const Tensor& mask) const;
    Tensor& operator=(const std::pair<Tensor, float>& masked_assignment);
    static Tensor from_condition(const Tensor& condition);
    Tensor operator>(float scalar) const;
    Tensor operator>(const Tensor& other) const;
    Tensor operator==(const Tensor& other) const;
    Tensor maxpool(int kernel_size, int stride, bool padding) const;
    Tensor avgpool(int kernel_size, int stride, bool padding) const;
    Tensor maxpool2d(int kernel_height, int kernel_width, int stride, bool padding) const;
    Tensor avgpool2d(int kernel_height, int kernel_width, int stride, bool padding) const;
    
    // Accessors
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    std::vector<int> shape() const { return shape_; }

private:
    std::vector<int> shape_;
    std::shared_ptr<float> data_; // Shared pointer for automatic memory management
    bool use_gpu_; // Whether to use GPU

    // Helper functions
    void allocate_memory();
    void free_memory();

    // CPU implementation of power
    Tensor power_cpu(float exponent) const;
};

#endif // TENSOR_H