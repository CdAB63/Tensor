#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <functional>
#include "cuda_kernels.h"
#include <iostream>

class Tensor {
    public:
        using EinsumOperation = std::function<Tensor(const Tensor&, const Tensor&)>;

        Tensor(const std::vector<int>& shape, bool use_gpu = false);
        ~Tensor();
    
        // Basic operations
        bool use_gpu() const;
        Tensor add(const Tensor& other, float alpha = 1.0f) const;
        float dot(const Tensor& other) const;
        Tensor conv2d(const Tensor& kernel, int stride, bool padding) const;
        Tensor power(float exponent) const;
        Tensor subtract(const Tensor& other) const;
        Tensor add_scaled(const Tensor& other, float alpha) const;
        Tensor multiply(const Tensor& other) const;
        Tensor divide(const Tensor& other) const;
        Tensor multiply_scalar(float scalar) const;
        Tensor sum(int axis) const;
        Tensor mean(int axis) const;
        Tensor max(int axis) const;
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
    
        // CPU implementation of convolution
        Tensor conv2d_cpu(const Tensor& kernel, int stride, bool padding) const;

	// CPU implementation of power
	Tensor power_cpu(float exponent) const;
    };

#endif // TENSOR_H
