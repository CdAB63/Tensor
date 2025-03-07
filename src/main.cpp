#include "Tensor.h"
#include <iostream>

void print_tensor(const Tensor& tensor, const std::string& name) {
    std::cout << name << ":\n";
    const float* data = tensor.data();
    std::vector<int> shape = tensor.shape();
    int size = 1;
    for (int dim : shape) size *= dim;

    for (int i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
        if ((i + 1) % shape[1] == 0) std::cout << "\n";
    }
    std::cout << "\n";
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
    float dot_product = A.dot(B);
    std::cout << "Dot product: " << dot_product << "\n\n";

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

    Tensor A3({2, 3}, false); // 2x3 matrix
    Tensor B2({3, 4}, false); // 3x4 matrix

    // Initialize D and E with dummy data
    for (int i = 0; i < 3; ++i) {
        D.data()[i] = static_cast<float>(i + 1);
        E.data()[i] = static_cast<float>(i + 1);
    }

    // Perform dot product using einsum
    Tensor F1 = D.einsum("i,i->", E);
    print_tensor(F, "Dot product using einsum");

    return 0;
}
