#include "Tensor.h"
#include <iostream>

void print_tensor(const Tensor& tensor, const std::string& name) {
    std::cout << name << ":\n";
    std::cout << "Shape: ";
    for (int dim : tensor.shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\n";

    const float* data = tensor.data();
    int size = 1;
    for (int dim : tensor.shape()) size *= dim;

    std::cout << "Data: ";
    for (int i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n\n";
}

Tensor matrix_multiplication(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 2 || B.shape().size() != 2 || A.shape()[1] != B.shape()[0]) {
        throw std::runtime_error("Invalid shapes for matrix multiplication");
    }

    int m = A.shape()[0];
    int n = A.shape()[1];
    int p = B.shape()[1];

    Tensor result({m, p}, A.use_gpu());

    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < p; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += A.data()[i * n + j] * B.data()[j * p + k];
            }
            result.data()[i * p + k] = sum;
        }
    }

    return result;
}

Tensor dot_product(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 1 || B.shape().size() != 1 || A.shape()[0] != B.shape()[0]) {
        throw std::runtime_error("Invalid shapes for dot product");
    }

    int n = A.shape()[0];
    Tensor result({1}, A.use_gpu());

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += A.data()[i] * B.data()[i];
    }
    result.data()[0] = sum;

    return result;
}

int main() {
    // Create two tensors (2x2)
    Tensor A({2, 2}, false); // CPU tensor
    Tensor B({2, 2}, false); // CPU tensor

    // Initialize A and B with dummy data
    A.data()[0] = 1.0f; A.data()[1] = 2.0f;
    A.data()[2] = 3.0f; A.data()[3] = 4.0f;

    B.data()[0] = 5.0f; B.data()[1] = 6.0f;
    B.data()[2] = 7.0f; B.data()[3] = 8.0f;

    // Print initial tensors
    print_tensor(A, "Tensor A");
    print_tensor(B, "Tensor B");

    // Test addition: C = A + B
    Tensor C = A.add(B, 1.0f);
    print_tensor(C, "C = A + B");

    // Test dot product
    float dot_result = A.dot(B);
    std::cout << "Dot product: " << dot_result << "\n\n";

    // Test subtraction: D = A - B
    Tensor D = A.subtract(B);
    print_tensor(D, "D = A - B");

    // Test scaled addition: E = A + 2.5 * B
    Tensor E = A.add_scaled(B, 2.5f);
    print_tensor(E, "E = A + 2.5 * B");

    // Test element-wise multiplication: F = A * B
    Tensor F = A.multiply(B);
    print_tensor(F, "F = A * B");

    // Test element-wise division: G = A / B
    Tensor G = A.divide(B);
    print_tensor(G, "G = A / B");

    // Test scalar multiplication: H = 3.0 * A
    Tensor H = A.multiply_scalar(3.0f);
    print_tensor(H, "H = 3.0 * A");

    // Test power operation: I = A^2
    Tensor I = A.power(2.0f);
    print_tensor(I, "I = A^2");

    // Create an input tensor (5x5 image with 3 channels)
    Tensor input({5, 5, 3}, false);

    // Create a kernel (3x3 kernel with 3 input channels and 2 filters)
    Tensor kernel({3, 3, 3, 2}, false);

    // Initialize input and kernel with dummy data
    for (int i = 0; i < 5 * 5 * 3; ++i) input.data()[i] = static_cast<float>(i) / 10.0f;
    for (int i = 0; i < 3 * 3 * 3 * 2; ++i) kernel.data()[i] = static_cast<float>(i) / 10.0f;

    // Perform convolution with stride 1 and padding
    Tensor output = input.conv2d(kernel, 1, true);

    // Print output shape
    std::cout << "Convolution output shape: " << output.shape()[0] << "x" << output.shape()[1] << "x" << output.shape()[2] << "\n\n";

    // Create a tensor (2x2x2)
    Tensor input1({2, 2, 2}, false);

    // Initialize input with dummy data
    for (int i = 0; i < 8; ++i) input1.data()[i] = static_cast<float>(i + 1);

    // Perform power operation
    Tensor result = input1.power(2.0f);

    // Print result
    std::cout << "Power operation result (2x2x2 tensor):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << result.data()[i] << " ";
    }
    std::cout << "\n";

    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, false);

    // Initialize A with dummy data
    for (int i = 0; i < 2 * 3 * 4; ++i) {
        A1.data()[i] = static_cast<float>(i);
    }

    // Test sum along axis 1
    Tensor sum_result = A1.sum(1);
    print_tensor(sum_result, "Sum along axis 1");

    // Test mean along axis 1
    Tensor mean_result = A1.mean(1);
    print_tensor(mean_result, "Mean along axis 1");

    // Test max along axis 1
    Tensor max_result = A1.max(1);
    print_tensor(max_result, "Max along axis 1");

    // Test min along axis 1
    Tensor min_result = A1.min(1);
    print_tensor(min_result, "Min along axis 1");

    // Test argmax along axis 1
    Tensor argmax_result = A1.argmax(1);
    print_tensor(argmax_result, "Argmax along axis 1");

    // Test argmin along axis 1
    Tensor argmin_result = A1.argmin(1);
    print_tensor(argmin_result, "Argmin along axis 1");

    Tensor A2({2, 2}, false);

    A2.data()[0] = 4.0f; A2.data()[1] = 1.0f;
    A2.data()[2] = 2.0f; A2.data()[3] = 3.0f;

    Tensor B1 = A2.matmul(A2);
    print_tensor(B1, "A2 * A2");

    Tensor invA = A2.inv();
    print_tensor(invA, "Inverse of A");

    Tensor transA = A2.transpose();
    print_tensor(transA, "Transpose of A");

    float detA = A2.det();
    std::cout << "Determinant of A: " << detA << "\n";

    auto [eigenvalue, eigenvector] = A2.eig();
    std::cout << "Dominant eigenvalue: " << eigenvalue << "\n";
    print_tensor(eigenvector, "Dominant eigenvector");

    // Matrix multiplication
    Tensor A3({2, 3}, false); // 2x3 matrix
    Tensor B3({3, 4}, false); // 3x4 matrix

    // Initialize A and B with dummy data
    for (int i = 0; i < 6; ++i) A3.data()[i] = static_cast<float>(i + 1);
    for (int i = 0; i < 12; ++i) B3.data()[i] = static_cast<float>(i + 1);

    // Perform matrix multiplication using einsum
    Tensor C3 = A3.einsum(std::function<Tensor(const Tensor&, const Tensor&)>(matrix_multiplication), B3);
    print_tensor(C3, "Matrix multiplication using einsum");

    // Dot product
    Tensor D3({3}, false); // 3-element vector
    Tensor E3({3}, false); // 3-element vector

    // Initialize D and E with dummy data
    for (int i = 0; i < 3; ++i) {
        D3.data()[i] = static_cast<float>(i + 1);
        E3.data()[i] = static_cast<float>(i + 1);
    }

    // Perform dot product using einsum
    Tensor F3 = D3.einsum(std::function<Tensor(const Tensor&, const Tensor&)>(dot_product), E3);
    print_tensor(F3, "Dot product using einsum");

    // Create a 2x3 tensor
    Tensor A4({2, 3}, false); // CPU tensor
    for (int i = 0; i < 6; ++i) A4.data()[i] = static_cast<float>(i + 1);
    print_tensor(A4, "Original Tensor A");

    // Test reshape
    Tensor reshaped = A4.reshape({3, 2});
    print_tensor(reshaped, "Reshaped Tensor A (3x2)");

    // Test flatten
    Tensor flattened = A4.flatten();
    print_tensor(flattened, "Flattened Tensor A");

    // Test expand_dims
    Tensor expanded = A4.expand_dims(1); // Add a dimension at axis 1
    print_tensor(expanded, "Expanded Tensor A (axis=1)");

    // Test squeeze
    Tensor squeezed = expanded.squeeze(); // Remove dimensions of size 1
    print_tensor(squeezed, "Squeezed Tensor (should match original A)");

    // Create another 2x3 tensor
    Tensor B4({2, 3}, false); // CPU tensor
    for (int i = 0; i < 6; ++i) B4.data()[i] = static_cast<float>(i + 7);
    print_tensor(B4, "Tensor B");

    // Test concat
    Tensor concatenated = A4.concat(B4, 0); // Concatenate along axis 0
    print_tensor(concatenated, "Concatenated Tensor (A and B along axis 0)");

    // Test stack
    Tensor stacked = Tensor::stack({A4, B4}, 0); // Stack along axis 0
    print_tensor(stacked, "Stacked Tensor (A and B along axis 0)");

    // Test permute
    Tensor permuted = A4.permute({1, 0}); // Swap dimensions
    print_tensor(permuted, "Permuted Tensor A (swapped dimensions)");

    // Additional tests for edge cases
    Tensor C4({1, 3, 1, 2}, false); // Tensor with singleton dimensions
    for (int i = 0; i < 6; ++i) C4.data()[i] = static_cast<float>(i + 1);
    print_tensor(C4, "Tensor C (1x3x1x2)");

    // Test squeeze on tensor with singleton dimensions
    Tensor squeezed_C = C4.squeeze();
    print_tensor(squeezed_C, "Squeezed Tensor C");

    // Test expand_dims on squeezed tensor
    Tensor expanded_C = squeezed_C.expand_dims(1);
    print_tensor(expanded_C, "Expanded Tensor C (axis=1)");

    return 0;
}
