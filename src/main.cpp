#include "Tensor.h"
#include <iostream>

void print_tensor(const Tensor& tensor, const std::string& name) {
    std::cout << name << ":\n";
    std::cout << "Shape: ";
    for (int dim : tensor.shape()) {
        std::cout << dim << " ";
    }
    std::cout << "\n";

    std::vector<float> data = tensor.get_data();
    std::cout << "Data: ";
    for (float val : data) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";
}

Tensor dot_product(const Tensor& A, const Tensor& B) {
    if (A.shape().size() != 1 || B.shape().size() != 1 || A.shape()[0] != B.shape()[0]) {
        throw std::runtime_error("Invalid shapes for dot product");
    }

    int n = A.shape()[0];
    Tensor result({1}, A.use_gpu());

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += A.get_data()[i] * B.get_data()[i];
    }
    result.load_data({sum});

    return result;
}

// Function to compare two tensors
bool compare_tensors(const Tensor& tensor1, const Tensor& tensor2, float epsilon = 1e-5) {
    // Check if shapes match
    if (tensor1.shape() != tensor2.shape()) {
        std::cerr << "Shape mismatch!\n";
        return false;
    }

    // Check if data matches (with epsilon tolerance for floating-point comparison)
    for (size_t i = 0; i < tensor1.size(); ++i) {
        if (std::abs(tensor1.data()[i] - tensor2.data()[i]) > epsilon) {
            std::cerr << "Data mismatch at index " << i << ": "
                      << tensor1.data()[i] << " != " << tensor2.data()[i] << "\n";
            return false;
        }
    }

    return true;
}

// TEST DOT PRODUCT <VECTOR>
bool test_dot(bool use_gpu) {
    std::cout << "Creating tensors A and B\n";
    Tensor A({4}, use_gpu); // Use GPU or CPU based on mode
    Tensor B({4}, use_gpu); // Use GPU or CPU based on mode
    std::cout << "Tensors A and B created\n";
    // Load data into tensors
    std::cout << "Loading data into tensor A\n";
    A.load_data({1.0f, 2.0f, 3.0f, 4.0f});
    std::cout << "Loading data into tensor B\n";
    B.load_data({4.0f, 3.0f, 2.0f, 1.0f});
    // Print initial tensors
    print_tensor(A, "Tensor A");
    print_tensor(B, "Tensor B");
    // Test dot product
    std::cout << "***** TESTING DOT PRODUCT *****\n";
    float dot_result = A.dot(B);
    std::cout << "Dot product: " << dot_result << "\n";
    if (dot_result != 20.0f) {
        std::cerr << "ERROR: result should be 20 but is " << dot_result << "\n\n";
        return false;
    } else {
        std::cout << "Dot product tested OK" << std::endl << std::endl;
    }

    return true;
}

// TEST D = A + B
bool test_a_plus_b(bool use_gpu) {
    std::cout << "***** TESTING ADD *****\n";
    Tensor A({2,2}, use_gpu); // Use GPU or CPU based on mode
    Tensor B({2,2}, use_gpu); // Use GPU or CPU based on mode
    A.load_data({1.0f, 2.0f, 3.0f, 4.0f});
    B.load_data({4.0f, 3.0f, 2.0f, 1.0f});
    Tensor THE_ADD = A.add(B);
    print_tensor(THE_ADD, "D = A + B");
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_data = {5.0, 5.0, 5.0, 5.0};
    if (THE_ADD.shape() == expected_shape && THE_ADD.get_data() == expected_data) {
        std::cout << "D = A + B tested OK" << "\n\n";
    } else {
        std::cerr << "D = A + B tested NOK" << "\n";
        print_tensor(A, "Tensor A");
        print_tensor(B, "Tensor B");
        print_tensor(THE_ADD, "THE_ADD (the sum) ");
        std::cerr << "Shape sould be {2, 2} and data {5, 5, 5, 5}";
        return false;
    }
    return true;
}

// TEST D = A - B
bool test_a_minus_b(bool use_gpu) {
    std::cout << "***** TESTING SUBRACT *****\n";
    Tensor A({2,2}, use_gpu); // Use GPU or CPU based on mode
    Tensor B({2,2}, use_gpu); // Use GPU or CPU based on mode
    A.load_data({1.0f, 2.0f, 3.0f, 4.0f});
    B.load_data({4.0f, 3.0f, 2.0f, 1.0f});
    Tensor SUBTRACT = A.subtract(B);
    print_tensor(SUBTRACT, "D = A + B");
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_data = {-3.0, -1.0, 1.0, 3.0};
    if (SUBTRACT.shape() == expected_shape && SUBTRACT.get_data() == expected_data) {
        std::cout << "D = A - B tested OK" << "\n\n";
    } else {
        std::cerr << "D = A - B tested NOK" << "\n";
        print_tensor(A, "Tensor A");
        print_tensor(B, "Tensor B");
        print_tensor(SUBTRACT, "SUBTRACT ");
        std::cerr << "Shape sould be {2, 2} and data {-3, -1, 1, 3}" << std::endl;
        return false;
    }
    return true;  
}

// Test scalled add
bool test_scaled_add(bool use_gpu) {
    std::cout << "***** TESTING D = A + alpha * B *****\n";
    Tensor A({2,2}, use_gpu); // Use GPU or CPU based on mode
    Tensor B({2,2}, use_gpu); // Use GPU or CPU based on mode
    float alpha = 0.5;
    A.load_data({1.0f, 2.0f, 3.0f, 4.0f});
    B.load_data({4.0f, 3.0f, 2.0f, 1.0f});
    Tensor SCALLED_ADD = A.add_scaled(B, alpha);
    print_tensor(SCALLED_ADD, "D = A + B");
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_data = {3.0, 3.5, 4.0, 4.5};
    if (SCALLED_ADD.shape() == expected_shape && SCALLED_ADD.get_data() == expected_data) {
        std::cout << "D = A + alpha * B tested OK" << "\n\n";
    } else {
        std::cerr << "D = A + alpha * B tested NOK" << "\n";
        print_tensor(A, "Tensor A");
        print_tensor(B, "Tensor B");
        print_tensor(SCALLED_ADD, "SCALLED_ADD ");
        std::cerr << "Shape sould be {2, 2} and data {-3, -1, 1, 3}" << std::endl;
        return false;
    }
    return true;  
}

int main(int argc, char* argv[]) {
    // Determine if we should use GPU or CPU
    bool use_gpu = false; // Default to CPU
    if (argc > 1) {
        std::string mode(argv[1]);
        if (mode == "GPU" || mode == "gpu") {
            use_gpu = true;
            std::cout << "Using GPU mode\n";
        } else if (mode == "CPU" || mode == "cpu") {
            use_gpu = false;
            std::cout << "Using CPU mode\n";
        } else {
            std::cerr << "Invalid argument. Use 'CPU' or 'GPU'.\n";
            return 1;
        }
    } else {
        std::cout << "No argument provided. Defaulting to CPU mode.\n";
    }

    if (!test_dot(use_gpu)) {
        std::cerr << "ERROR test_dot failed\n";
        return false;
    }

    // Test add: D = A + B
    if (!test_a_plus_b(use_gpu)) {
        std::cerr << "ERROR test_a_plus_b failed\n";
        return false;
    }

    // Test subtraction: D = A - B
    if (!test_a_minus_b(use_gpu)) {
        std::cerr << "ERROR test_a_minus_b failed\n";
        return false;
    }

    // Test scaled add c = a + x*b
    if (!test_scaled_add(use_gpu)) {
        std::cerr<< "ERROR test_scaled_add failed\n";
        return false;
    }

    // Test element-wise multiplication: F = A * B
    Tensor A({2}, use_gpu);
    Tensor B({2}, use_gpu);
    A.load_data({1.0, 2.0});
    B.load_data({2.0, 1.0});
    std::cout << "***** TESTING MULTIPLY *****\n";
    Tensor F = A.multiply(B);
    print_tensor(F, "F = A * B");

    // Test element-wise division: G = A / B
    std::cout << "***** TESTING DIVIDE *****\n";
    Tensor G = A.divide(B);
    print_tensor(G, "G = A / B");

    // Test scalar multiplication: H = 3.0 * A
    std::cout << "***** TESTING MULTIPLY_SCALAR *****\n";
    Tensor H = A.multiply_scalar(3.0f);
    print_tensor(H, "H = 3.0 * A");

    // Test power operation: I = A^2
    std::cout << "***** TESTING POWER *****\n";
    Tensor I = A.power(2.0f);
    print_tensor(I, "I = A^2");

    // Create a 1D input tensor (batch_size=1, in_channels=1, length=5)
    Tensor input1d({1, 1, 5}, use_gpu);
    input1d.load_data({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    // Create a 1D kernel (kernel_size=3, in_channels=1, out_channels=1)
    Tensor kernel1d({3, 1, 1}, use_gpu);
    kernel1d.load_data({1.0f, 2.0f, 3.0f});

    // Perform 1D convolution
    std::cout << "***** TESTING 1D CONVOLUTION *****\n";
    Tensor output1d = input1d.conv1d(kernel1d, 1, true);
    print_tensor(output1d, "1D Convolution Output");

    // Create an input tensor (5x5 image with 3 channels)
    Tensor input2d({5, 5, 3}, use_gpu);
    std::vector<float> input2d_data(5 * 5 * 3);
    for (int i = 0; i < 5 * 5 * 3; ++i) input2d_data[i] = static_cast<float>(i) / 10.0f;
    input2d.load_data(input2d_data);

    // Create a kernel (3x3 kernel with 3 input channels and 2 filters)
    Tensor kernel2d({3, 3, 3, 2}, use_gpu);
    std::vector<float> kernel2d_data(3 * 3 * 3 * 2);
    for (int i = 0; i < 3 * 3 * 3 * 2; ++i) kernel2d_data[i] = static_cast<float>(i) / 10.0f;
    kernel2d.load_data(kernel2d_data);

    // Perform convolution with stride 1 and padding
    Tensor output2d = input2d.conv2d(kernel2d, 1, true);

    // Print output shape
    std::cout << "Convolution output shape: " << output2d.shape()[0] << "x" << output2d.shape()[1] << "x" << output2d.shape()[2] << "\n\n";
    print_tensor(output2d, "2D Convolution Output");

    // Create a 3D input tensor (batch_size=1, in_channels=1, depth=3, height=3, width=3)
    Tensor input3d({1, 1, 3, 3, 3}, use_gpu);
    std::vector<float> input3d_data(27);
    for (int i = 0; i < 27; ++i) input3d_data[i] = static_cast<float>(i + 1);
    input3d.load_data(input3d_data);

    // Create a 3D kernel (kernel_depth=2, kernel_height=2, kernel_width=2, in_channels=1, out_channels=1)
    Tensor kernel3d({2, 2, 2, 1, 1}, use_gpu);
    std::vector<float> kernel3d_data(8);
    for (int i = 0; i < 8; ++i) kernel3d_data[i] = static_cast<float>(i + 1);
    kernel3d.load_data(kernel3d_data);

    // Perform 3D convolution
    Tensor output3d = input3d.conv3d(kernel3d, 1, true);
    print_tensor(output3d, "3D Convolution Output");

    // Create a tensor (2x2x2)
    Tensor input1({2, 2, 2}, use_gpu);
    std::vector<float> input1_data(8);
    for (int i = 0; i < 8; ++i) input1_data[i] = static_cast<float>(i + 1);
    input1.load_data(input1_data);

    // Perform power operation
    Tensor result = input1.power(2.0f);

    // Print result
    std::cout << "Power operation result (2x2x2 tensor):\n";
    std::vector<float> result_data = result.get_data();
    for (float val : result_data) {
        std::cout << val << " ";
    }
    std::cout << "\n";

    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);

    // Test sum along axis 1
    std::cout << "***** TEST SUM *****\n";
    Tensor sum_result = A1.sum(1);
    print_tensor(sum_result, "Sum along axis 1");

    // Test mean along axis 1
    std::cout << "***** TEST MEAN ALONG AXIS *****\n";
    Tensor mean_result = A1.mean(1);
    print_tensor(mean_result, "Mean along axis 1");

    // Test max along axis 1
    std::cout << "***** TEST MAX ALONG AXIS *****\n";
    Tensor max_result = A1.max(1);
    print_tensor(max_result, "Max along axis 1");

    // Test min along axis 1
    std::cout << "***** TEST MIN ALONG AXIS *****\n";
    Tensor min_result = A1.min(1);
    print_tensor(min_result, "Min along axis 1");

    // Test argmax along axis 1
    std::cout << "***** TEST ARGMAX ALONG AXIS *****\n";
    Tensor argmax_result = A1.argmax(1);
    print_tensor(argmax_result, "Argmax along axis 1");

    // Test argmin along axis 1
    std::cout << "***** TEST ARGMIN ALONG AXIS *****\n";
    Tensor argmin_result = A1.argmin(1);
    print_tensor(argmin_result, "Argmin along axis 1");

    Tensor A2({2, 2}, use_gpu);
    A2.load_data({4.0f, 1.0f, 2.0f, 3.0f});

    // Test matmul
    std::cout << "***** TEST MATMUL *****\n";
    Tensor B1 = A2.matmul(A2);
    print_tensor(B1, "A2 * A2");

    // Test inv
    std::cout << "***** TEST INV *****\n";
    Tensor invA = A2.inv();
    print_tensor(invA, "Inverse of A");

    // Test transpose
    std::cout << "***** TEST TRANSPOSE *****\n";
    Tensor transA = A2.transpose();
    print_tensor(transA, "Transpose of A");

    // Test determinant
    std::cout << "***** TEST DET *****\n";
    float detA = A2.det();
    std::cout << "Determinant of A: " << detA << "\n";

    // Test eigen vector and eigen value
    std::cout << "***** TEST EIG *****\n";
    auto [eigenvalue, eigenvector] = A2.eig();
    std::cout << "Dominant eigenvalue: " << eigenvalue << "\n";
    print_tensor(eigenvector, "Dominant eigenvector");

    // Dot product
    Tensor D3({3}, use_gpu); // 3-element vector
    std::vector<float> D3_data(3);
    for (int i = 0; i < 3; ++i) D3_data[i] = static_cast<float>(i + 1);
    D3.load_data(D3_data);

    Tensor E3({3}, use_gpu); // 3-element vector
    std::vector<float> E3_data(3);
    for (int i = 0; i < 3; ++i) E3_data[i] = static_cast<float>(i + 1);
    E3.load_data(E3_data);

    // Perform dot product using einsum
    std::cout << "***** TEST DOT EINSUM *****\n";
    Tensor F3 = D3.einsum(std::function<Tensor(const Tensor&, const Tensor&)>(dot_product), E3);
    print_tensor(F3, "Dot product using einsum");

    // Create a 2x3 tensor
    std::cout << "***** TEST RESHAPE *****\n";
    Tensor A4({2, 3}, use_gpu); // Use GPU or CPU based on mode
    std::vector<float> A4_data(6);
    for (int i = 0; i < 6; ++i) A4_data[i] = static_cast<float>(i + 1);
    A4.load_data(A4_data);
    print_tensor(A4, "Original Tensor A");

    // Test reshape
    Tensor reshaped = A4.reshape({3, 2});
    print_tensor(reshaped, "Reshaped Tensor A (3x2)");

    // Test flatten
    std::cout << "***** TEST FLATTEN *****\n";
    Tensor flattened = A4.flatten();
    print_tensor(flattened, "Flattened Tensor A");

    // Test expand_dims
    std::cout << "***** TEST EXPAND_DIMS *****\n";
    Tensor expanded = A4.expand_dims(1); // Add a dimension at axis 1
    print_tensor(expanded, "Expanded Tensor A (axis=1)");

    // Test squeeze
    std::cout << "***** TEST SQUEEZE *****\n";
    Tensor squeezed = expanded.squeeze(); // Remove dimensions of size 1
    print_tensor(squeezed, "Squeezed Tensor (should match original A)");

    // Create another 2x3 tensor
    Tensor B4({2, 3}, use_gpu); // Use GPU or CPU based on mode
    std::vector<float> B4_data(6);
    for (int i = 0; i < 6; ++i) B4_data[i] = static_cast<float>(i + 7);
    B4.load_data(B4_data);
    print_tensor(B4, "Tensor B");

    // Test concat
    std::cout << "***** TEST CONCAT *****\n";
    Tensor concatenated = A4.concat(B4, 0); // Concatenate along axis 0
    print_tensor(concatenated, "Concatenated Tensor (A and B along axis 0)");

    // Test stack
    std::cout << "***** TEST STACK *****\n";
    Tensor stacked = Tensor::stack({A4, B4}, 0); // Stack along axis 0
    print_tensor(stacked, "Stacked Tensor (A and B along axis 0)");

    // Test permute
    std::cout << "***** TEST PERMUTE *****\n";
    Tensor permuted = A4.permute({1, 0}); // Swap dimensions
    print_tensor(permuted, "Permuted Tensor A (swapped dimensions)");

    // Additional tests for edge cases
    Tensor C4({1, 3, 1, 2}, use_gpu); // Tensor with singleton dimensions
    std::vector<float> C4_data(6);
    for (int i = 0; i < 6; ++i) C4_data[i] = static_cast<float>(i + 1);
    C4.load_data(C4_data);
    print_tensor(C4, "Tensor C (1x3x1x2)");

    // Test squeeze on tensor with singleton dimensions
    std::cout << "***** TEST SQUEEZE *****\n";
    Tensor squeezed_C = C4.squeeze();
    print_tensor(squeezed_C, "Squeezed Tensor C");

    // Test expand_dims on squeezed tensor
    std::cout << "***** TEST EXPAND_DIMS *****\n";
    Tensor expanded_C = squeezed_C.expand_dims(1);
    print_tensor(expanded_C, "Expanded Tensor C (axis=1)");

    // Create tensors
    Tensor A5({2, 3}, use_gpu); // 2x3 tensor
    Tensor B5({3}, use_gpu);    // 3-element vector

    // Initialize tensors
    std::vector<float> A5_data(6);
    std::vector<float> B5_data(3);
    for (int i = 0; i < 6; ++i) A5_data[i] = static_cast<float>(i + 7);
    for (int i = 0; i < 3; ++i) B5_data[i] = static_cast<float>(i);
    A5.load_data(A5_data);
    B5.load_data(B5_data);

    // Print tensors
    print_tensor(A5, "Tensor A");
    print_tensor(B5, "Tensor B");

    // Test broadcasting
    std::cout << "***** TEST BROADCAST_TENSORS *****\n";
    auto [A_broadcasted, B_broadcasted] = Tensor::broadcast_tensors(A5, B5);
    print_tensor(A_broadcasted, "Broadcasted Tensor A");
    print_tensor(B_broadcasted, "Broadcasted Tensor B");

    // Test element-wise comparison
    std::cout << "***** TEST > *****\n";
    Tensor greater = A5 > B5;
    print_tensor(greater, "A > B");

    std::cout << "***** TEST == *****\n";
    Tensor equal = A4 == B4;
    print_tensor(equal, "A == B");

    // Create a tensor
    Tensor A6({2, 3}, use_gpu); // 2x3 tensor
    std::vector<float> A6_data(6);
    for (int i = 0; i < 6; ++i) A6_data[i] = static_cast<float>(i + 1);
    A6.load_data(A6_data);

    // Print original tensor
    print_tensor(A6, "Original Tensor A");

    // Compare with a scalar
    std::cout << "***** TEST MASK > float *****\n";
    Tensor mask = A6 > 3.0f;
    print_tensor(mask, "Mask (A > 3)");

    // Compare with another tensor (broadcasting)
    Tensor B6({3}, use_gpu); // 3-element vector
    std::vector<float> B6_data(3);
    for (int i = 0; i < 3; ++i) B6_data[i] = static_cast<float>(i + 2);
    B6.load_data(B6_data);

    std::cout << "***** TEST MASK A > B *****\n";
    Tensor mask2 = A6 > B6;
    print_tensor(mask2, "Mask (A > B)");

    // Create a 3D input tensor (batch_size=1, channels=1, length=6)
    Tensor inputmp({1, 1, 6}, use_gpu);
    std::vector<float> inputmp_data(6);
    for (int i = 0; i < 6; ++i) inputmp_data[i] = static_cast<float>(i + 1);
    inputmp.load_data(inputmp_data);

    // Print original tensor
    print_tensor(inputmp, "Original Tensor");

    // Test max pooling
    std::cout << "***** TEST MAXPOOL *****\n";
    Tensor max_pooled = inputmp.maxpool(2, 2, use_gpu);
    print_tensor(max_pooled, "Max Pooled Tensor");

    // Test average pooling
    std::cout << "***** TEST AVGPOOL *****\n";
    Tensor avg_pooled = inputmp.avgpool(2, 2, use_gpu);
    print_tensor(avg_pooled, "Average Pooled Tensor");

    // TESTING REPEAT
    std::cout << "***** TEST REPEAT *****\n";
    // Create a sample tensor
    std::cout << "***** TEST REPEAT CREATING A SAMPLE TENSOR *****\n";
    std::vector<int> shape = {2, 3}; // 2x3 tensor
    Tensor A7(shape, false); // Use CPU
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::copy(data, data + 6, A7.data()); // Fill tensor with data

    // Print ori ginal tensor
    print_tensor(A7, "Original Tensor A");

    // Test Tensor::repeat along axis 0 with 2 repeats
    std::cout << "***** TEST REPEAT ALONG AXIS 0 (2 REPEATS) *****\n";
    Tensor A_repeated_axis0 = A7.repeat(0, 2);
    print_tensor(A_repeated_axis0, "Repeated Tensor (Axis 0)");

    // Expected result for axis 0 repeat
    Tensor expected_axis0({4, 3}, false);
    float expected_data_axis0[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                   1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::copy(expected_data_axis0, expected_data_axis0 + 12, expected_axis0.data());

    // Verify result
    if (compare_tensors(A_repeated_axis0, expected_axis0)) {
        std::cout << "Test passed: Repeated Tensor (Axis 0) matches expected result!\n";
    } else {
        std::cerr << "Test failed: Repeated Tensor (Axis 0) does not match expected result!\n";
    }

    // Test Tensor::repeat along axis 1 with 3 repeats
    std::cout << "\n***** TEST REPEAT ALONG AXIS 1 (3 REPEATS) *****\n";
    Tensor A_repeated_axis1 = A7.repeat(1, 3);
    print_tensor(A_repeated_axis1, "Repeated Tensor (Axis 1)");

    // Expected result for axis 1 repeat
    Tensor expected_axis1({2, 9}, false);
    float expected_data_axis1[] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f};
    std::copy(expected_data_axis1, expected_data_axis1 + 18, expected_axis1.data());

    // Verify result
    if (compare_tensors(A_repeated_axis1, expected_axis1)) {
        std::cout << "Test passed: Repeated Tensor (Axis 1) matches expected result!\n";
    } else {
        std::cerr << "Test failed: Repeated Tensor (Axis 1) does not match expected result!\n";
    }

    // Test Tensor::repeat along axis 1 with 1 repeat (no change)
    std::cout << "\n***** TEST REPEAT ALONG AXIS 1 (1 REPEAT) *****\n";
    Tensor A_repeated_axis1_nochange = A7.repeat(1, 1);
    print_tensor(A_repeated_axis1_nochange, "Repeated Tensor (Axis 1, 1 Repeat)");

    // Verify result
    if (compare_tensors(A_repeated_axis1_nochange, A7)) {
        std::cout << "Test passed: Repeated Tensor (Axis 1, 1 Repeat) matches original tensor!\n";
    } else {
        std::cerr << "Test failed: Repeated Tensor (Axis 1, 1 Repeat) does not match original tensor!\n";
    }

    std::cout << "***** TEST MASKED ASSIGNMENT *****\n";
    Tensor the_tensor({2, 3}, false); // CPU tensor
    Tensor the_mask({2, 3}, false);   // CPU mask

    // Initialize data
    the_tensor.load_data({1, 2, 3, 4, 5, 6});
    the_mask.load_data({0, 1, 0, 1, 0, 1});

    // Perform masked assignment
    the_tensor = {the_mask, 10.0f};

    // Check the result
    std::vector<float> expected = {1, 10, 3, 10, 5, 10};
    assert(the_tensor.get_data() == expected);

    std::cout << "Masked assignment test passed!" << std::endl;

    return 0;
}