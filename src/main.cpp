#include "Tensor.h"
#include <iostream>

// Helper functions
std::string format_indices(const std::vector<int>& indices) {
    std::string s = "(";
    for (size_t i = 0; i < indices.size(); ++i) {
        s += std::to_string(indices[i]);
        if (i != indices.size()-1) s += ", ";
    }
    return s + ")";
}

void print_vector(const std::vector<int>& v) {
    std::cerr << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cerr << v[i];
        if (i != v.size()-1) std::cerr << ", ";
    }
    std::cerr << "]\n";
}

bool compare_data(const std::vector<float>& a, const std::vector<float>& b, float epsilon) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

void print_shape(const std::vector<int>& shape) {
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << shape[i];
    }
    std::cerr << "]\n";
}

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

bool compare_tensors(const Tensor& tensor1, const Tensor& tensor2, float epsilon = 1e-5) {
    // Check if shapes match
    if (tensor1.shape() != tensor2.shape()) {
        std::cerr << "Shape mismatch!\n";
        return false;
    }

    // Get host-accessible data copies
    std::vector<float> data1 = tensor1.get_data();
    std::vector<float> data2 = tensor2.get_data();

    // Check if data matches (with epsilon tolerance)
    for (size_t i = 0; i < tensor1.size(); ++i) {
        if (std::abs(data1[i] - data2[i]) > epsilon) {
            std::cerr << "Data mismatch at index " << i << ": "
                      << data1[i] << " != " << data2[i] << "\n";
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

// Test multiply
bool test_multiply(bool use_gpu) {
    std::cout << "***** TESTING MULTIPLY *****\n";
    Tensor A({2, 2}, use_gpu);
    Tensor B({2, 2}, use_gpu);
    A.load_data({1.0, 2.0, 3.0, 4.0});
    B.load_data({4.0, 3.0, 2.0, 1.0});
    Tensor D = A.multiply(B);
    print_tensor(D, "F = A * B");
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_data = {4.0, 6.0, 6.0, 4.0};
    if (D.shape() == expected_shape && D.get_data() == expected_data) {
        std::cout << "D = A * B tested OK" << "\n\n";
    } else { 
        std::cerr << "D = A * B teste NOK" << "\n";
        print_tensor(A, "Tensor A");
        print_tensor(B, "Tensor B");
        print_tensor(D, "Tensor D");
        std::cerr << "Shape should be (2, 2) and data (4.0, 6.0, 6.0, 4.0)\n";
        return false;
    }
    return true;
}

// Test divide
bool test_divide(bool use_gpu) {
    std::cout << "***** TESTING DIVIDE *****\n";
    Tensor A({2, 2}, use_gpu);
    Tensor B({2, 2}, use_gpu);
    A.load_data({1.0, 3.0, 3.0, 4.0});
    B.load_data({4.0, 3.0, 2.0, 1.0});
    Tensor D = A.divide(B);
    print_tensor(D, "F = A / B");
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_data = {0.25, 1.0, 1.5, 4.0};
    if (D.shape() == expected_shape && D.get_data() == expected_data) {
        std::cout << "D = A / B tested OK" << "\n\n";
    } else { 
        std::cerr << "D = A / B teste NOK" << "\n";
        print_tensor(A, "Tensor A");
        print_tensor(B, "Tensor B");
        print_tensor(D, "Tensor D");
        std::cerr << "Shape should be (2, 2) and data (0.25, 1.0, 1.5, 4.0)\n";
        return false;
    }
    return true;
}

bool test_multiply_scalar(bool use_gpu) {
    std::cout << "***** TESTING MULTIPLY SCALAR *****\n";
    Tensor A({2,2}, use_gpu);
    A.load_data({1.0, 2.0, 3.0, 4.0});
    Tensor C = A.multiply_scalar(2.5);
    print_tensor(C, "C = 2.5 * A");
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_data = {2.5, 5.0, 7.5, 10.0};
    if (C.shape() == expected_shape && C.get_data() == expected_data) {
        std::cout << "C = 2.5 * A tested OK\n\n";
    } else {
        std::cerr << "C = 2.5 * A tested NOK\n";
        print_tensor(A, "Tensor A");
        print_tensor(C, "Tensor C");
        std::cerr << "Shape should be {2, 2} and data {2.5, 5.0, 7.5, 10.0}\n";
        return false;
    }
    return true;
}

// Test power
bool test_power(bool use_gpu) {
    std::cout << "***** TESTING POWER OPERATION *****\n";
    Tensor A({2,2}, use_gpu);
    A.load_data({1.0, 2.0, 3.0, 4.0});
    Tensor C = A.power(2);
    print_tensor(C, "C = A.power(2)");
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_result = {1.0, 4.0, 9.0, 16.0};
    if (C.shape() == expected_shape && C.get_data() == expected_result) {
        std::cout << "C = A.power(2) tested OK\n\n";
    } else {
        std::cerr << "C = A.power(2) tested NOK\n";
        print_tensor(A, "Tensor A");
        print_tensor(C, "Tensor C");
        std::cerr << "Shape should be (2, 2) and data (1, 4, 9, 16)\n";
        return false;
    }
    return true;
}

// Test conv1d
bool test_conv1d(bool use_gpu) {
    std::cout << "***** TESTING 1D CONVOLUTION *****\n";
    // Create a 1D input tensor (batch_size=1, in_channels=1, length=5)
    Tensor input1d({1, 1, 10}, use_gpu);
    input1d.load_data({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    // Create a 1D kernel (kernel_size=3, in_channels=1, out_channels=1)
    Tensor kernel1d({3, 1, 1}, use_gpu);
    kernel1d.load_data({1.0f, 2.0f, 3.0f});
    // Perform 1D convolution
    Tensor output1d = input1d.conv1d(kernel1d, 1, true);
    print_tensor(output1d, "1D Convolution Output");
    std::vector<int> expected_shape = {1, 1, 10};
    std::vector<float> expected_result = {8, 14, 20, 26, 32, 38, 44, 50, 56, 29};
    if (output1d.shape() == expected_shape && output1d.get_data() == expected_result) {
        std::cout << "C = A.conv1d() tested OK\n\n";
    } else {
        std::cerr << "C = A.conv1d() tested NOK\n";
        print_tensor(input1d, "The vector");
        print_tensor(kernel1d, "The kernel");
        print_tensor(output1d, "The convolution");
        std::cerr << "Shape shoud be (1, 1, 8) and data (8, 14, 20, 26,32, 38, 44, 50, 56, 29)\n";
        return false;
    }
    return true;
}

// Test conv2d
bool test_conv2d(bool use_gpu) {
    std::cout << "***** TESTING 2D CONVOLUTION *****\n";
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
    std::vector<int> expected_shape = {5, 5, 2};
    if (output2d.shape() == expected_shape) {
        std::cout << "conv2d = input2d.conv2d() tested OK\n\n";
    } else {
        std::cerr << "conv2d = input2d.conv2d() tested NOK\n";
        return false;
    }
    return true;
}

// Test conv3d
bool test_conv3d(bool use_gpu) {
    std::cout << "***** TESTING 3D CONVOLUTION *****\n";
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
    std::vector<int> expected_shape = {1, 1, 2, 2, 2};
    if (output3d.shape() == expected_shape) {
        std::cout << "conv3d = input3d.conv3d() tested OK\n\n";
    } else {
        std::cerr << "conv3d = input3d.conv3d() tested NOK\n";
        return false;
    }
    return true;
}

// Test sum along axis
bool test_sum_along_axis(bool use_gpu) {
    std::cout << "***** TEST SUM ALONG AXIS *****\n";
    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);
    print_tensor(A1, "Input data");
    // Test sum along axis 1
    Tensor sum_result = A1.sum(1);
    print_tensor(sum_result, "Sum along axis 1");
    // Correct expected shape and data
    std::vector<int> expected_shape = {2, 4};
    std::vector<float> expected_result = {12, 15, 18, 21, 48, 51, 54, 57};
    if (sum_result.shape() == expected_shape && sum_result.get_data() == expected_result) {
        std::cout << "sum_result = A1.sum() tested OK\n\n";
        return true;
    } else {
        std::cerr << "sum_result = A1.sum() tested NOK\n";
        print_tensor(sum_result, "SUM RESULT");
        return false;
    }
}

bool test_mean_along_axis(bool use_gpu) {
    std::cout << "***** TEST MEAN ALONG AXIS *****\n";
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(24);
    for (int i = 0; i < 24; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);

    Tensor mean_result = A1.mean(1);
    print_tensor(mean_result, "Mean along axis 1");

    std::vector<int> expected_shape = {2, 4};
    std::vector<float> expected_result = {
        4.0f, 5.0f, 6.0f, 7.0f,
        16.0f, 17.0f, 18.0f, 19.0f
    };

    if (mean_result.shape() == expected_shape && mean_result.get_data() == expected_result) {
        std::cout << "mean tested OK\n\n";
        return true;
    } else {
        std::cerr << "mean result tested NOK\n";
        print_tensor(mean_result, "MEAN RESULT");
        return false;
    }
}

// Test max (scalar version)
bool test_max(bool use_gpu) {
    std::cout << "***** TEST MAX *****\n";
    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);  // Fill tensor with sequential values
    A1.load_data(A1_data);

    // Calculate max using the Tensor::max() method
    float max_value = A1.max();

    // Expected max value (since the tensor holds {0, 1, ..., 23})
    float expected_max_value = 23.0f;

    // Verify the result
    if (max_value == expected_max_value) {
        std::cout << "Max value tested OK: " << max_value << "\n\n";
        return true;
    } else {
        std::cerr << "Max value tested NOK: expected " << expected_max_value << ", got " << max_value << "\n\n";
        return false;
    }
}

// Test max along axis
bool test_max_along_axis(bool use_gpu) {
    std::cout << "***** TEST MAX ALONG AXIS *****\n";
    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);
    // Test max along axis 1
    Tensor max_result = A1.max(1);
    print_tensor(max_result, "Max along axis 1");
    // Define expected shape and data
    std::vector<int> expected_shape = {2, 4};
    std::vector<float> expected_result = {8, 9, 10, 11, 20, 21, 22, 23};
    // Check if the computed max_result matches
    if (max_result.shape() == expected_shape && max_result.get_data() == expected_result) {
        std::cout << "max along axis 1 tested OK\n\n";
        return true;
    } else {
        std::cerr << "max along axis 1 tested NOK\n";
        print_tensor(max_result, "MAX RESULT");
        return false;
    }
}

// Test scalar minimum value in a tensor
bool test_min(bool use_gpu) {
    std::cout << "***** TEST MIN *****\n";
    
    // Create a tensor and load values
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);

    // Compute the minimum value (scalar)
    float result = A1.min();
    
    // Expected result: the minimum value in the tensor (which is 0 in this case)
    float expected_result = 0.0f;

    if (result == expected_result) {
        std::cout << "min tested OK\n\n";
        return true;
    } else {
        std::cerr << "min tested NOK: expected " << expected_result << " but got " << result << "\n";
        return false;
    }
}

// Test minimum along a specified axis
bool test_min_along_axis(bool use_gpu) {
    std::cout << "***** TEST MIN ALONG AXIS *****\n";
    
    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);

    // Test min along axis 1
    Tensor min_result = A1.min(1);
    print_tensor(min_result, "Min along axis 1");

    // Define expected shape and data
    std::vector<int> expected_shape = {2, 4};
    std::vector<float> expected_result = {0, 1, 2, 3, 12, 13, 14, 15};

    // Check if the computed min_result matches the expected values
    if (min_result.shape() == expected_shape && min_result.get_data() == expected_result) {
        std::cout << "min along axis 1 tested OK\n\n";
        return true;
    } else {
        std::cerr << "min along axis 1 tested NOK\n";
        print_tensor(min_result, "MIN RESULT");
        return false;
    }
}

// Test argmax
bool test_argmax(bool use_gpu) {
    std::cout << "***** TEST ARGMAX ALONG AXIS *****\n";
    
    // Create 2x3x4 tensor with known values
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(24);
    for (int i = 0; i < 24; ++i) A1_data[i] = i; // Values 0-23
    A1.load_data(A1_data);

    // Test argmax along axis 1 (middle dimension)
    Tensor argmax_result = A1.argmax(1);
    print_tensor(argmax_result, "Argmax along axis 1");

    // Expected results:
    // For each of the 2 batches and 4 positions:
    // - In 0th batch: max indices [8,9,10,11] → all index 2
    // - In 1st batch: max indices [20,21,22,23] → all index 2
    std::vector<int> expected_shape = {2, 4};
    std::vector<float> expected_result(8, 2.0f); // 8 elements, all 2.0

    // Validate
    if (argmax_result.shape() == expected_shape && 
        argmax_result.get_data() == expected_result) {
        std::cout << "argmax along axis 1 tested OK\n\n";
        return true;
    } else {
        std::cerr << "argmax along axis 1 tested NOK\n";
        print_tensor(argmax_result, "ARGMAX RESULT");
        return false;
    }
}

// Test argmin
bool test_argmin(bool use_gpu) {
    std::cout << "***** TEST ARGMIN ALONG AXIS *****\n";
    
    // Test Case 1: 2x3x4 Tensor
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(24);
    for (int i = 0; i < 24; ++i) A1_data[i] = i; // Values 0-23
    A1.load_data(A1_data);

    // Test argmin along axis 1 (middle dimension)
    Tensor argmin_result = A1.argmin(1);
    print_tensor(argmin_result, "Argmin along axis 1 (3D Tensor)");

    // Expected results for 2x3x4 tensor:
    // - For each batch (2), the minimum in axis 1 (3 elements) will be at index 0
    std::vector<int> expected_shape_3D = {2, 4};
    std::vector<float> expected_result_3D(8, 0.0f); // All indices 0
    
    bool test1_ok = (argmin_result.shape() == expected_shape_3D) && 
                    (argmin_result.get_data() == expected_result_3D);

    // Test Case 2: 2x2 Tensor
    Tensor A2({2, 2}, use_gpu);
    A2.load_data({4.0f, 1.0f, 2.0f, 3.0f});
    Tensor argmin_result_2D = A2.argmin(1);
    print_tensor(argmin_result_2D, "Argmin along axis 1 (2D Tensor)");

    // Expected results for 2x2 tensor:
    // Row 0: [4,1] → argmin=1, Row 1: [2,3] → argmin=0
    std::vector<int> expected_shape_2D = {2};
    std::vector<float> expected_result_2D = {1.0f, 0.0f};
    
    bool test2_ok = (argmin_result_2D.shape() == expected_shape_2D) && 
                    (argmin_result_2D.get_data() == expected_result_2D);

    if (test1_ok && test2_ok) {
        std::cout << "argmin tested OK\n\n";
        return true;
    } else {
        if (!test1_ok) std::cerr << "3D argmin failed!\n";
        if (!test2_ok) std::cerr << "2D argmin failed!\n";
        return false;
    }
}

// Test matmul
bool test_matmul(bool use_gpu) {
    std::cout << "***** TEST MATMUL *****\n";

    // Create two input tensors (2x2)
    Tensor A({2, 2}, use_gpu);
    Tensor B({2, 2}, use_gpu);

    // Load data into the tensors
    A.load_data({1.0f, 2.0f, 3.0f, 4.0f}); // Matrix: [[1, 2], [3, 4]]
    B.load_data({2.0f, 0.0f, 1.0f, 2.0f}); // Matrix: [[2, 0], [1, 2]]

    // Perform matrix multiplication: C = A * B
    Tensor C = A.matmul(B);
    print_tensor(C, "A * B");

    // Expected result: [[4, 4], [10, 8]]
    std::vector<int> expected_shape = {2, 2};
    std::vector<float> expected_result = {4.0f, 4.0f, 10.0f, 8.0f};

    bool test_ok = (C.shape() == expected_shape) && (C.get_data() == expected_result);

    if (test_ok) {
        std::cout << "Matmul tested OK\n\n";
        return true;
    } else {
        std::cerr << "Matmul test failed!\n";
        return false;
    }
}

bool test_inv(bool use_gpu) {
    std::cout << "***** TEST INV *****\n";

    // Create reference CPU tensor
    Tensor A_cpu({2, 2}, false);
    A_cpu.load_data({4.0f, 7.0f, 2.0f, 6.0f});
    Tensor invA_cpu = A_cpu.inv();

    // Compute inverse on GPU/CPU
    Tensor A({2, 2}, use_gpu);
    A.load_data({4.0f, 7.0f, 2.0f, 6.0f});
    Tensor invA = A.inv();

    // If GPU was used, copy result to CPU for comparison
    Tensor invA_host = invA;
    if (use_gpu) {
        invA_host = Tensor(invA.shape(), false); // Create CPU tensor
        invA_host.load_data(invA.get_data()); // Copies GPU→CPU
    }

    print_tensor(invA_host, "Inverse of A (CPU-converted)");

    // Compare CPU tensors
    bool test_ok = compare_tensors(invA_host, invA_cpu, 1e-5f);

    if (test_ok) {
        std::cout << "Matrix inversion tested OK\n\n";
        return true;
    } else {
        std::cerr << "Matrix inversion test failed!\n";
        return false;
    }
}

// Test transpose
bool test_transpose(bool use_gpu) {
    std::cout << "***** TEST TRANSPOSE *****\n";
    // Create a tensor (4x4)
    Tensor A1({4, 4}, use_gpu);
    std::vector<float> A1_data(4 * 4);
    for (int i = 0; i < 4 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);
    // Test transpose
    Tensor transA = A1.transpose();
    print_tensor(transA, "Transpose of A");

    // Expected shape remains (4,4) for square matrix
    std::vector<int> expected_shape = {4, 4};
    if (transA.shape() != expected_shape) {
        std::cerr << "Error: Transposed tensor shape is incorrect. Expected (4, 4), got ("
                  << transA.shape()[0] << ", " << transA.shape()[1] << ")\n";
        return false;
    }

    // Generate expected transposed data (columns become rows)
    std::vector<float> expected_data;
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            expected_data.push_back(A1_data[row * 4 + col]);
        }
    }

    // Check if transposed data matches expected values
    std::vector<float> transposed_data = transA.get_data();
    if (transposed_data != expected_data) {
        std::cerr << "Error: Transposed data does not match expected values.\n";
        std::cerr << "Expected data: ";
        for (float val : expected_data) std::cerr << val << " ";
        std::cerr << "\nActual data:   ";
        for (float val : transposed_data) std::cerr << val << " ";
        std::cerr << "\n";
        return false;
    }

    std::cout << "Transpose tested OK\n\n";
    return true;
}

// Test determinant
bool test_determinant(bool use_gpu) {
    std::cout << "***** TEST DET *****\n";
    // Create a tensor (2x2)
    Tensor A1({2, 2}, use_gpu);
    A1.load_data({4.0f, 7.0f, 2.0f, 6.0f});
    // Test determinant
    float detA = A1.det();
    std::cout << "Determinant of A: " << detA << "\n";
    
    // Expected determinant: (4*6) - (7*2) = 24 - 14 = 10
    float expected_det = 10.0f;
    float epsilon = 1e-5;
    bool test_ok = std::abs(detA - expected_det) < epsilon;

    if (test_ok) {
        std::cout << "Determinant tested OK\n\n";
        return true;
    } else {
        std::cerr << "Determinant tested NOK: expected " << expected_det << ", got " << detA << "\n\n";
        return false;
    }
}

bool test_eigen(bool use_gpu) {
    std::cout << "***** TEST EIG *****\n";
    
    // Test matrix: [[4, 1], [1, 4]]
    Tensor A({2, 2}, use_gpu);
    A.load_data({4.0f, 1.0f,
                 1.0f, 4.0f});

    auto [eigenvalue, eigenvector] = A.eig();
    
    std::cout << "Dominant eigenvalue: " << eigenvalue << "\n";
    print_tensor(eigenvector, "Dominant eigenvector");

    // Verify eigenvalue (should be ~5.0)
    const float expected_eigenvalue = 5.0f;
    const float epsilon = 1e-2f;
    
    // Verify eigenvector property: A*v ≈ λ*v
    Tensor Av = A.matmul(eigenvector);
    Tensor lambda_v = eigenvector.multiply_scalar(eigenvalue);
    
    bool property_ok = compare_tensors(Av.flatten(), lambda_v.flatten(), epsilon);
    bool eigenvalue_ok = std::abs(eigenvalue - expected_eigenvalue) < epsilon;

    if (eigenvalue_ok && property_ok) {
        std::cout << "Eigen test OK\n\n";
        return true;
    }

    if (!eigenvalue_ok) {
        std::cerr << "Eigenvalue mismatch! Expected " << expected_eigenvalue 
                  << ", got " << eigenvalue << "\n";
    }
    if (!property_ok) {
        std::cerr << "Eigenvector property A*v != λ*v failed!\n";
        print_tensor(Av, "A*v");
        print_tensor(lambda_v, "λ*v");
    }

    return false;

}

bool test_dot_product_einsum(bool use_gpu) {
    try {
        std::cout << "***** TEST DOT EINSUM *****\n";
        
        // Create vectors
        Tensor D3({3}, use_gpu);
        Tensor E3({3}, use_gpu);
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        D3.load_data(data);
        E3.load_data(data);

        // Perform dot product using einsum
        Tensor F3 = D3.einsum(std::function<Tensor(const Tensor&, const Tensor&)>(dot_product), E3);
        print_tensor(F3, "Dot product using einsum");

        // Verify results
        const float expected_dot = 14.0f; // 1*1 + 2*2 + 3*3
        const float epsilon = 1e-5f;

        // Check output shape
        if (F3.shape() != std::vector<int>{1}) {
            std::cerr << "Shape mismatch! Expected [1], got [";
            for (size_t i = 0; i < F3.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << F3.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // Check output value
        float result = F3.get_data()[0];
        if (std::abs(result - expected_dot) > epsilon) {
            std::cerr << "Dot product mismatch! Expected " << expected_dot 
                      << ", got " << result << "\n";
            return false;
        }

        std::cout << "Dot product einsum test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in dot product einsum test: " << e.what() << "\n";
        return false;
    }
}

// Test reshape
bool test_reshape(bool use_gpu) {
    std::cout << "***** TEST RESHAPE *****\n";
    // Create original tensor
    Tensor A4({2, 3}, use_gpu);
    std::vector<float> A4_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    A4.load_data(A4_data);
    print_tensor(A4, "Original Tensor A");
    try {
        std::cout << "***** TEST RESHAPE *****\n";

        // Perform reshape
        Tensor reshaped = A4.reshape({3, 2});
        print_tensor(reshaped, "Reshaped Tensor A (3x2)");

        // Verify results
        const float epsilon = 1e-5f;
        
        // 1. Check output shape
        if (reshaped.shape() != std::vector<int>{3, 2}) {
            std::cerr << "Shape mismatch! Expected [3, 2], got [";
            for (size_t i = 0; i < reshaped.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << reshaped.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }
        // 2. Check data integrity
        std::vector<float> original_data = A4.get_data();
        std::vector<float> reshaped_data = reshaped.get_data();
        
        if (original_data.size() != reshaped_data.size()) {
            std::cerr << "Data size mismatch after reshape! Original: " 
                      << original_data.size() << ", Reshaped: "
                      << reshaped_data.size() << "\n";
            return false;
        }

        bool data_match = true;
        for (size_t i = 0; i < original_data.size(); ++i) {
            if (std::abs(original_data[i] - reshaped_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i << ": "
                          << original_data[i] << " vs " << reshaped_data[i] << "\n";
                data_match = false;
            }
        }

        if (!data_match) {
            std::cerr << "Reshape data verification failed!\n";
            return false;
        }

        // 3. Additional check: reshape back to original
        Tensor reshaped_back = reshaped.reshape(A4.shape());
        if (!compare_tensors(A4, reshaped_back, epsilon)) {
            std::cerr << "Round-trip reshape verification failed!\n";
            return false;
        }

        std::cout << "Reshape test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in reshape test: " << e.what() << "\n";
        return false;
    }
}

// Test flatten
bool test_flatten(bool use_gpu) {
    try {
        std::cout << "***** TEST FLATTEN *****\n";
        
        // Create original tensor
        Tensor A4({2, 3}, use_gpu);
        std::vector<float> A4_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        A4.load_data(A4_data);
        print_tensor(A4, "Original Tensor A");

        // Perform flatten operation
        Tensor flattened = A4.flatten();
        print_tensor(flattened, "Flattened Tensor A");

        // Verify results
        const float epsilon = 1e-5f;

        // 1. Check output shape
        if (flattened.shape() != std::vector<int>{6}) {
            std::cerr << "Shape mismatch! Expected [6], got [";
            for (size_t i = 0; i < flattened.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << flattened.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check data integrity
        std::vector<float> original_data = A4.get_data();
        std::vector<float> flattened_data = flattened.get_data();
        
        if (original_data.size() != flattened_data.size()) {
            std::cerr << "Data size mismatch! Original: " 
                      << original_data.size() << ", Flattened: "
                      << flattened_data.size() << "\n";
            return false;
        }

        bool data_match = true;
        for (size_t i = 0; i < original_data.size(); ++i) {
            if (std::abs(original_data[i] - flattened_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i << ": "
                          << original_data[i] << " vs " << flattened_data[i] << "\n";
                data_match = false;
            }
        }

        if (!data_match) {
            std::cerr << "Flatten data verification failed!\n";
            return false;
        }

        // 3. Additional check: verify element order
        std::vector<float> expected_flattened = {1,2,3,4,5,6};
        for (size_t i = 0; i < expected_flattened.size(); ++i) {
            if (std::abs(flattened_data[i] - expected_flattened[i]) > epsilon) {
                std::cerr << "Element order mismatch at index " << i << ": "
                          << flattened_data[i] << " vs expected " 
                          << expected_flattened[i] << "\n";
                return false;
            }
        }

        // 4. Round-trip test
        Tensor unflattened = flattened.reshape(A4.shape());
        if (!compare_tensors(A4, unflattened, epsilon)) {
            std::cerr << "Round-trip flatten/reshape verification failed!\n";
            return false;
        }

        std::cout << "Flatten test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in flatten test: " << e.what() << "\n";
        return false;
    }
}

// Test expand_dims
bool test_expand_dims(bool use_gpu) {
    try {
        std::cout << "***** TEST EXPAND DIMS *****\n";
        
        // Create original tensor
        Tensor A4({2, 3}, use_gpu);
        std::vector<float> A4_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        A4.load_data(A4_data);
        print_tensor(A4, "Original Tensor A");

        // Perform expand_dims operation
        Tensor expanded = A4.expand_dims(1);
        print_tensor(expanded, "Expanded Tensor A (axis=1)");

        // Verify results
        const float epsilon = 1e-5f;

        // 1. Check output shape
        const std::vector<int> expected_shape = {2, 1, 3};
        if (expanded.shape() != expected_shape) {
            std::cerr << "Shape mismatch! Expected [2, 1, 3], got [";
            for (size_t i = 0; i < expanded.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << expanded.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check data integrity
        std::vector<float> original_data = A4.get_data();
        std::vector<float> expanded_data = expanded.get_data();
        
        if (original_data.size() != expanded_data.size()) {
            std::cerr << "Data size mismatch! Original: " 
                      << original_data.size() << ", Expanded: "
                      << expanded_data.size() << "\n";
            return false;
        }

        bool data_match = true;
        for (size_t i = 0; i < original_data.size(); ++i) {
            if (std::abs(original_data[i] - expanded_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i << ": "
                          << original_data[i] << " vs " << expanded_data[i] << "\n";
                data_match = false;
            }
        }

        if (!data_match) {
            std::cerr << "Expand_dims data verification failed!\n";
            return false;
        }

        // 3. Check dimension semantics
        if (expanded.shape()[1] != 1) {
            std::cerr << "Expanded dimension should be size 1, got "
                      << expanded.shape()[1] << "\n";
            return false;
        }

        // 4. Round-trip test
        Tensor squeezed = expanded.squeeze();
        if (!compare_tensors(A4, squeezed, epsilon)) {
            std::cerr << "Round-trip expand_dims/squeeze verification failed!\n";
            return false;
        }

        std::cout << "Expand_dims test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in expand_dims test: " << e.what() << "\n";
        return false;
    }
}

// Test squeeze
bool test_squeeze(bool use_gpu) {
    try {
        std::cout << "***** TEST SQUEEZE *****\n";
        
        // Create original tensor
        Tensor A4({2, 3}, use_gpu);
        std::vector<float> A4_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        A4.load_data(A4_data);
        print_tensor(A4, "Original Tensor A");

        // Add then remove singleton dimension
        Tensor expanded = A4.expand_dims(1);
        Tensor squeezed = expanded.squeeze();
        print_tensor(squeezed, "Squeezed Tensor");

        // Verify results
        const float epsilon = 1e-5f;

        // 1. Check output shape matches original
        if (squeezed.shape() != A4.shape()) {
            std::cerr << "Shape mismatch! Expected [2, 3], got [";
            for (size_t i = 0; i < squeezed.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << squeezed.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check data integrity
        std::vector<float> original_data = A4.get_data();
        std::vector<float> squeezed_data = squeezed.get_data();
        
        if (original_data.size() != squeezed_data.size()) {
            std::cerr << "Data size mismatch! Original: " 
                      << original_data.size() << ", Squeezed: "
                      << squeezed_data.size() << "\n";
            return false;
        }

        bool data_match = true;
        for (size_t i = 0; i < original_data.size(); ++i) {
            if (std::abs(original_data[i] - squeezed_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i << ": "
                          << original_data[i] << " vs " << squeezed_data[i] << "\n";
                data_match = false;
            }
        }

        if (!data_match) {
            std::cerr << "Squeeze data verification failed!\n";
            return false;
        }

        // 3. Verify squeeze removed ALL singleton dimensions
        for (int dim : squeezed.shape()) {
            if (dim == 1) {
                std::cerr << "Found residual singleton dimension in squeezed tensor\n";
                return false;
            }
        }

        // 4. Additional test: squeeze already minimal tensor
        Tensor should_not_change = A4.squeeze();
        if (!compare_tensors(A4, should_not_change, epsilon)) {
            std::cerr << "Squeeze modified tensor with no singleton dimensions!\n";
            return false;
        }

        std::cout << "Squeeze test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in squeeze test: " << e.what() << "\n";
        return false;
    }
}


// Test concat
bool test_concat(bool use_gpu) {
    try {
        std::cout << "***** TEST CONCAT *****\n";
        
        // Create tensors
        Tensor A({2, 3}, use_gpu);
        std::vector<float> A_data = {1,2,3,4,5,6};
        A.load_data(A_data);
        print_tensor(A, "Tensor A");

        Tensor B({2, 3}, use_gpu);
        std::vector<float> B_data = {7,8,9,10,11,12};
        B.load_data(B_data);
        print_tensor(B, "Tensor B");

        // Perform concatenation
        const int axis = 0;
        Tensor concatenated = A.concat(B, axis);
        print_tensor(concatenated, "Concatenated Tensor (axis 0)");

        // Verify results
        const float epsilon = 1e-5f;
        const std::vector<int> expected_shape = {4, 3};
        const std::vector<float> expected_data = 
            {1,2,3,4,5,6,7,8,9,10,11,12};

        // 1. Check output shape
        if (concatenated.shape() != expected_shape) {
            std::cerr << "Shape mismatch! Expected [4, 3], got [";
            for (size_t i = 0; i < concatenated.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << concatenated.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check data integrity
        std::vector<float> concat_data = concatenated.get_data();
        bool data_ok = true;
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(concat_data[i] - expected_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i 
                          << ": expected " << expected_data[i]
                          << ", got " << concat_data[i] << "\n";
                data_ok = false;
            }
        }
        if (!data_ok) return false;

        // 3. Verify original tensors unchanged
        Tensor A_copy({2, 3}, use_gpu);
        A_copy.load_data(A_data);
        Tensor B_copy({2, 3}, use_gpu);
        B_copy.load_data(B_data);
        
        if (!compare_tensors(A, A_copy, epsilon) ||
            !compare_tensors(B, B_copy, epsilon)) {
            std::cerr << "Original tensors modified during concatenation!\n";
            return false;
        }

        std::cout << "Concat test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in concat test: " << e.what() << "\n";
        return false;
    }
}

// Test stack
bool test_stack(bool use_gpu) {
    try {
        std::cout << "***** TEST STACK *****\n";
        
        // Create tensors
        Tensor A({2, 3}, use_gpu);
        std::vector<float> A_data = {1,2,3,4,5,6};
        A.load_data(A_data);
        print_tensor(A, "Tensor A");

        Tensor B({2, 3}, use_gpu);
        std::vector<float> B_data = {7,8,9,10,11,12};
        B.load_data(B_data);
        print_tensor(B, "Tensor B");

        // Perform stacking
        const int axis = 0;
        Tensor stacked = Tensor::stack({A, B}, axis);
        print_tensor(stacked, "Stacked Tensor");

        // Verify results
        const float epsilon = 1e-5f;
        const std::vector<int> expected_shape = {2, 2, 3};
        const std::vector<float> expected_data = 
            {1,2,3,4,5,6,7,8,9,10,11,12};

        // 1. Check output shape
        if (stacked.shape() != expected_shape) {
            std::cerr << "Shape mismatch! Expected [2, 2, 3], got [";
            for (size_t i = 0; i < stacked.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << stacked.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check data integrity
        std::vector<float> stacked_data = stacked.get_data();
        bool data_ok = true;
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(stacked_data[i] - expected_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i 
                          << ": expected " << expected_data[i]
                          << ", got " << stacked_data[i] << "\n";
                data_ok = false;
            }
        }
        if (!data_ok) return false;

        // 3. Verify original tensors unchanged
        Tensor A_copy({2, 3}, use_gpu);
        A_copy.load_data(A_data);
        Tensor B_copy({2, 3}, use_gpu);
        B_copy.load_data(B_data);
        
        if (!compare_tensors(A, A_copy, epsilon) ||
            !compare_tensors(B, B_copy, epsilon)) {
            std::cerr << "Original tensors modified during stacking!\n";
            return false;
        }

        std::cout << "Stack test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in stack test: " << e.what() << "\n";
        return false;
    }
}

// Test permute
bool test_permute(bool use_gpu) {
    try {
        std::cout << "***** TEST PERMUTE *****\n";
        
        // Create original tensor
        Tensor A({2, 3}, use_gpu);
        std::vector<float> A_data = {1,2,3,4,5,6};
        A.load_data(A_data);
        print_tensor(A, "Original Tensor A");

        // Perform permutation
        Tensor permuted = A.permute({1, 0});
        print_tensor(permuted, "Permuted Tensor");

        // Verify results
        const float epsilon = 1e-5f;
        const std::vector<int> expected_shape = {3, 2};
        const std::vector<float> expected_data = 
            {1,4,2,5,3,6}; // Transposed version

        // 1. Check output shape
        if (permuted.shape() != expected_shape) {
            std::cerr << "Shape mismatch! Expected [3, 2], got [";
            for (size_t i = 0; i < permuted.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << permuted.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check data integrity
        std::vector<float> permuted_data = permuted.get_data();
        bool data_ok = true;
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(permuted_data[i] - expected_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i 
                          << ": expected " << expected_data[i]
                          << ", got " << permuted_data[i] << "\n";
                data_ok = false;
            }
        }
        if (!data_ok) return false;

        // 3. Verify original tensor unchanged
        Tensor A_copy({2, 3}, use_gpu);
        A_copy.load_data(A_data);
        if (!compare_tensors(A, A_copy, epsilon)) {
            std::cerr << "Original tensor modified during permutation!\n";
            return false;
        }

        // 4. Round-trip test
        Tensor permuted_back = permuted.permute({1, 0});
        if (!compare_tensors(A, permuted_back, epsilon)) {
            std::cerr << "Round-trip permutation verification failed!\n";
            return false;
        }

        std::cout << "Permute test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in permute test: " << e.what() << "\n";
        return false;
    }
}

// Test edge conditions
bool test_edge_squeeze_expand(bool use_gpu) {
    try {
        const float epsilon = 1e-5f;
        std::cout << "***** TEST EDGE CASES (SQUEEZE/EXPAND_DIMS) *****\n";

        // Test 1: Squeeze with singleton dimensions
        std::cout << "-- Testing squeeze --\n";
        Tensor C4({1, 3, 1, 2}, use_gpu);
        std::vector<float> C4_data = {1,2,3,4,5,6};
        C4.load_data(C4_data);
        print_tensor(C4, "Original Tensor C (1x3x1x2)");

        Tensor squeezed_C = C4.squeeze();
        print_tensor(squeezed_C, "Squeezed Tensor C");

        // Verify squeeze results
        const std::vector<int> expected_squeezed_shape = {3, 2};
        const std::vector<float> expected_squeezed_data = C4_data; // Same data, new shape
        
        // 1. Check squeezed shape
        if (squeezed_C.shape() != expected_squeezed_shape) {
            std::cerr << "Squeeze shape mismatch! Expected [3, 2], got [";
            print_shape(squeezed_C.shape());
            return false;
        }

        // 2. Check squeezed data matches original data
        if (!compare_data(squeezed_C.get_data(), expected_squeezed_data, epsilon)) {
            std::cerr << "Squeezed data mismatch!\n";
            return false;
        }

        // Test 2: Expand_dims
        std::cout << "-- Testing expand_dims --\n";
        Tensor expanded_C = squeezed_C.expand_dims(1);
        print_tensor(expanded_C, "Expanded Tensor C (axis=1)");

        // Verify expand_dims results
        const std::vector<int> expected_expanded_shape = {3, 1, 2};
        const std::vector<float> expected_expanded_data = C4_data; // Same data, new shape
        
        // 1. Check expanded shape
        if (expanded_C.shape() != expected_expanded_shape) {
            std::cerr << "Expand_dims shape mismatch! Expected [3, 1, 2], got [";
            print_shape(expanded_C.shape());
            return false;
        }

        // 2. Check expanded data matches original data
        if (!compare_data(expanded_C.get_data(), expected_expanded_data, epsilon)) {
            std::cerr << "Expanded data mismatch!\n";
            return false;
        }

        // Test 3: Verify original tensor unchanged
        Tensor C4_copy({1, 3, 1, 2}, use_gpu);
        C4_copy.load_data(C4_data);
        if (!compare_tensors(C4, C4_copy, epsilon)) {
            std::cerr << "Original tensor modified during operations!\n";
            return false;
        }

        std::cout << "Edge case tests (squeeze/expand_dims) OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in edge case tests: " << e.what() << "\n";
        return false;
    }
}

// Test broadcast
bool test_broadcast(bool use_gpu) {
    try {
        std::cout << "***** TEST BROADCAST_TENSORS *****\n";
        
        // Create tensors
        Tensor A({2, 3}, use_gpu);
        std::vector<float> A_data = {7,8,9,10,11,12};
        A.load_data(A_data);
        print_tensor(A, "Tensor A");

        Tensor B({3}, use_gpu);
        std::vector<float> B_data = {0,1,2};
        B.load_data(B_data);
        print_tensor(B, "Tensor B");

        // Perform broadcasting
        auto [A_broadcasted, B_broadcasted] = Tensor::broadcast_tensors(A, B);
        print_tensor(A_broadcasted, "Broadcasted Tensor A");
        print_tensor(B_broadcasted, "Broadcasted Tensor B");

        // Verify results
        const float epsilon = 1e-5f;
        const std::vector<int> expected_A_shape = {2, 3};
        const std::vector<int> expected_B_shape = {2, 3};
        const std::vector<float> expected_B_data = 
            {0,1,2,0,1,2}; // Broadcasted version of B

        // 1. Check output shapes
        if (A_broadcasted.shape() != expected_A_shape) {
            std::cerr << "A shape mismatch! Expected [2, 3], got [";
            for (size_t i = 0; i < A_broadcasted.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << A_broadcasted.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        if (B_broadcasted.shape() != expected_B_shape) {
            std::cerr << "B shape mismatch! Expected [2, 3], got [";
            for (size_t i = 0; i < B_broadcasted.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << B_broadcasted.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check broadcasted B data
        std::vector<float> B_bc_data = B_broadcasted.get_data();
        bool data_ok = true;
        for (size_t i = 0; i < expected_B_data.size(); ++i) {
            if (std::abs(B_bc_data[i] - expected_B_data[i]) > epsilon) {
                std::cerr << "B data mismatch at index " << i 
                          << ": expected " << expected_B_data[i]
                          << ", got " << B_bc_data[i] << "\n";
                data_ok = false;
            }
        }
        if (!data_ok) return false;

        // 3. Verify original tensors unchanged
        Tensor A_copy({2, 3}, use_gpu);
        A_copy.load_data(A_data);
        Tensor B_copy({3}, use_gpu);
        B_copy.load_data(B_data);
        
        if (!compare_tensors(A, A_copy, epsilon) ||
            !compare_tensors(B, B_copy, epsilon)) {
            std::cerr << "Original tensors modified during broadcasting!\n";
            return false;
        }

        std::cout << "Broadcast test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in broadcast test: " << e.what() << "\n";
        return false;
    }
}

bool test_comparisons(bool use_gpu) {
    try {
        std::cout << "***** TEST COMPARISON OPERATORS *****\n";
        const float epsilon = 1e-5f;

        // ==================== > OPERATOR TEST ====================
        {
            std::cout << "Testing > operator...\n";
            // Create tensors with mixed true/false cases
            Tensor A({2, 3}, use_gpu);
            std::vector<float> A_data = {3,1,5,2,6,4};  // 2x3
            A.load_data(A_data);
            print_tensor(A, "Tensor A");

            Tensor B({3}, use_gpu);    
            std::vector<float> B_data = {2,0,3};        // 3-element vector
            B.load_data(B_data);
            print_tensor(B, "Tensor B");

            // Perform operation
            Tensor result = A > B;
            print_tensor(result, "A > B result");

            // Expected results (broadcasted comparison)
            const std::vector<int> expected_shape = {2, 3};
            const std::vector<float> expected_data = {1,1,1,0,1,1};

            // 1. Check shape
            if (result.shape() != expected_shape) {
                std::cerr << "> operator shape mismatch! Expected [2, 3], got [";
                print_shape(result.shape());
                return false;
            }

            // 2. Check data
            std::vector<float> result_data = result.get_data();
            bool data_ok = true;
            for (size_t i = 0; i < expected_data.size(); ++i) {
                if (std::abs(result_data[i] - expected_data[i]) > epsilon) {
                    std::cerr << "> operator data mismatch at " << i 
                              << ": expected " << expected_data[i]
                              << ", got " << result_data[i] << "\n";
                    data_ok = false;
                }
            }
            if (!data_ok) return false;

            // 3. Verify originals unchanged
            Tensor A_copy({2, 3}, use_gpu);
            A_copy.load_data(A_data);
            Tensor B_copy({3}, use_gpu);
            B_copy.load_data(B_data);
            if (!compare_tensors(A, A_copy, epsilon) || 
                !compare_tensors(B, B_copy, epsilon)) {
                std::cerr << "Originals modified during > operation!\n";
                return false;
            }
        }

        // ==================== == OPERATOR TEST ====================
        {
            std::cout << "Testing == operator...\n";
            // Create tensors with partial matches
            Tensor A({2, 3}, use_gpu);
            std::vector<float> A_data = {1,2,3,4,5,6};
            A.load_data(A_data);
            print_tensor(A, "Tensor A");

            Tensor B({2, 3}, use_gpu);
            std::vector<float> B_data = {1,0,3,4,5,7};  // Partial matches
            B.load_data(B_data);
            print_tensor(B, "Tensor B");

            // Perform operation
            Tensor result = A == B;
            print_tensor(result, "A == B result");

            // Expected results
            const std::vector<int> expected_shape = {2, 3};
            const std::vector<float> expected_data = {1,0,1,1,1,0};

            // 1. Check shape
            if (result.shape() != expected_shape) {
                std::cerr << "== operator shape mismatch! Expected [2, 3], got [";
                print_shape(result.shape());
                return false;
            }

            // 2. Check data
            std::vector<float> result_data = result.get_data();
            bool data_ok = true;
            for (size_t i = 0; i < expected_data.size(); ++i) {
                if (std::abs(result_data[i] - expected_data[i]) > epsilon) {
                    std::cerr << "== operator data mismatch at " << i 
                              << ": expected " << expected_data[i]
                              << ", got " << result_data[i] << "\n";
                    data_ok = false;
                }
            }
            if (!data_ok) return false;

            // 3. Verify originals unchanged
            Tensor A_copy({2, 3}, use_gpu);
            A_copy.load_data(A_data);
            Tensor B_copy({2, 3}, use_gpu);
            B_copy.load_data(B_data);
            if (!compare_tensors(A, A_copy, epsilon) || 
                !compare_tensors(B, B_copy, epsilon)) {
                std::cerr << "Originals modified during == operation!\n";
                return false;
            }
        }

        // ==================== NEGATIVE TEST CASES ====================
        {
            std::cout << "Testing negative cases...\n";
            // Test shape mismatch that should throw
            Tensor C({2, 3}, use_gpu);
            Tensor D({2, 4}, use_gpu);
            try {
                auto result = C > D;  // Should throw
                std::cerr << "Failed to catch shape mismatch in > operator!\n";
                return false;
            } catch (const std::runtime_error&) {
                // Expected error
            }

            // Test empty tensor (scalar with initialized data)
            Tensor E({}, use_gpu);
            E.load_data({5.0f}); // Initialize with data
            Tensor F({}, use_gpu);
            F.load_data({5.0f}); // Initialize with same data
            try {
                auto result = E == F;
                if (result.size() != 1 || result.get_data()[0] != 1.0f) {
                    print_tensor(result, "RESULT FOR EMPTY TENSOR");
                    std::cerr << "Empty tensor comparison failed!\n";
                    return false;
                }
            } catch (...) {
                std::cerr << "Unexpected error with empty tensors!\n";
                return false;
            }
        }

        std::cout << "All comparison tests OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in comparison test: " << e.what() << "\n";
        return false;
    }
}

// Test masked comparisons
bool test_masked_comparisons(bool use_gpu) {
    try {
        const float epsilon = 1e-5f;
        std::cout << "***** TEST MASKED COMPARISONS *****\n";

        // ==================== A > SCALAR TEST ====================
        Tensor A({2, 3}, use_gpu);
        std::vector<float> A_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        A.load_data(A_data);
        print_tensor(A, "Tensor A");

        Tensor mask_scalar = A > 3.0f;
        print_tensor(mask_scalar, "A > 3.0 result");

        // Expected: [[0,0,0], [1,1,1]]
        std::vector<float> expected_scalar = {0,0,0,1,1,1};
        if (mask_scalar.shape() != std::vector<int>{2, 3}) {
            std::cerr << "Shape mismatch for scalar comparison\n";
            return false;
        }
        std::vector<float> result_scalar = mask_scalar.get_data();
        for (size_t i = 0; i < expected_scalar.size(); ++i) {
            if (std::abs(result_scalar[i] - expected_scalar[i]) > epsilon) {
                std::cerr << "Data mismatch at " << i << ": "
                          << result_scalar[i] << " vs " << expected_scalar[i] << "\n";
                return false;
            }
        }

        // ==================== A > TENSOR (BROADCAST) TEST ====================
        Tensor B({3}, use_gpu);
        std::vector<float> B_data = {2.0f, 3.0f, 4.0f};
        B.load_data(B_data);
        print_tensor(B, "Tensor B");

        Tensor mask_tensor = A > B;
        print_tensor(mask_tensor, "A > B result");

        // Expected: [[0,0,0], [1,1,1]]
        std::vector<float> expected_tensor = {0,0,0,1,1,1};
        if (mask_tensor.shape() != std::vector<int>{2, 3}) {
            std::cerr << "Shape mismatch for tensor comparison\n";
            return false;
        }
        std::vector<float> result_tensor = mask_tensor.get_data();
        for (size_t i = 0; i < expected_tensor.size(); ++i) {
            if (std::abs(result_tensor[i] - expected_tensor[i]) > epsilon) {
                std::cerr << "Data mismatch at " << i << ": "
                          << result_tensor[i] << " vs " << expected_tensor[i] << "\n";
                return false;
            }
        }

        // ==================== VERIFY ORIGINAL TENSORS UNCHANGED ====================
        std::vector<float> A_post = A.get_data();
        for (size_t i = 0; i < A_data.size(); ++i) {
            if (std::abs(A_post[i] - A_data[i]) > epsilon) {
                std::cerr << "Original tensor A modified during comparison!\n";
                return false;
            }
        }

        std::vector<float> B_post = B.get_data();
        for (size_t i = 0; i < B_data.size(); ++i) {
            if (std::abs(B_post[i] - B_data[i]) > epsilon) {
                std::cerr << "Original tensor B modified during comparison!\n";
                return false;
            }
        }

        std::cout << "All masked comparison tests passed!\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return false;
    }
}

// Test maxpool
bool test_maxpool(bool use_gpu) {
    try {
        std::cout << "***** TEST MAXPOOL *****\n";
        
        // Create input tensor (batch_size=1, channels=1, length=6)
        Tensor inputmp({1, 1, 6}, use_gpu);
        std::vector<float> inputmp_data = {1, 2, 3, 4, 5, 6};
        inputmp.load_data(inputmp_data);
        print_tensor(inputmp, "Original Tensor");

        // Perform max pooling with kernel=2, stride=2, no padding
        Tensor max_pooled = inputmp.maxpool(2, 2, false);
        print_tensor(max_pooled, "Max Pooled Tensor");

        // Expected results
        const std::vector<int> expected_shape = {1, 1, 3};
        const std::vector<float> expected_data = {2.0f, 4.0f, 6.0f};
        const float epsilon = 1e-5f;

        // 1. Check output shape
        if (max_pooled.shape() != expected_shape) {
            std::cerr << "Shape mismatch! Expected [1, 1, 3], got [";
            for (size_t i = 0; i < max_pooled.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << max_pooled.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check output data
        std::vector<float> pooled_data = max_pooled.get_data();
        bool data_ok = true;
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(pooled_data[i] - expected_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i 
                          << ": expected " << expected_data[i]
                          << ", got " << pooled_data[i] << "\n";
                data_ok = false;
            }
        }
        if (!data_ok) return false;

        // 3. Verify original tensor is unchanged
        Tensor original_copy({1, 1, 6}, use_gpu);
        original_copy.load_data(inputmp_data);
        if (!compare_tensors(inputmp, original_copy, epsilon)) {
            std::cerr << "Original tensor was modified during maxpool operation!\n";
            return false;
        }

        std::cout << "Maxpool test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in maxpool test: " << e.what() << "\n";
        return false;
    }
}

// Test repeat
bool test_repeat(bool use_gpu) {
    try {
        std::cout << "***** TEST REPEAT *****\n";
        const float epsilon = 1e-5f;

        // Create original tensor
        Tensor A7({2, 3}, use_gpu);
        std::vector<float> A7_data = {1, 2, 3, 4, 5, 6};
        A7.load_data(A7_data);
        print_tensor(A7, "Original Tensor");

        // Test 1: Repeat axis 0 (rows) with 2 repeats
        {
            std::cout << "TESTING AXIS 0 REPEAT (2x)...\n";
            Tensor result = A7.repeat(0, 2);
            print_tensor(result, "Axis 0 Repeated");

            // Create expected tensor
            Tensor expected({4, 3}, use_gpu);
            expected.load_data({1,2,3,4,5,6,1,2,3,4,5,6});

            if (!compare_tensors(result, expected, epsilon)) {
                std::cerr << "Axis 0 repeat failed!\n";
                return false;
            }
        }

        // Test 2: Repeat axis 1 (columns) with 3 repeats
        {
            std::cout << "TESTING AXIS 1 REPEAT (3x)...\n";
            Tensor result = A7.repeat(1, 3);
            print_tensor(result, "Axis 1 Repeated");

            // Create expected tensor
            Tensor expected({2, 9}, use_gpu);
            expected.load_data({
                1,2,3,1,2,3,1,2,3,
                4,5,6,4,5,6,4,5,6
            });

            if (!compare_tensors(result, expected, epsilon)) {
                std::cerr << "Axis 1 repeat failed!\n";
                return false;
            }
        }

        // Test 3: Verify original unchanged
        {
            Tensor original_copy({2, 3}, use_gpu);
            original_copy.load_data(A7_data);
            if (!compare_tensors(A7, original_copy, epsilon)) {
                std::cerr << "Original tensor modified!\n";
                return false;
            }
        }

        std::cout << "All repeat tests passed!\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in repeat test: " << e.what() << "\n";
        return false;
    }
}

bool test_avgpool(bool use_gpu) {
    try {
        std::cout << "***** TEST AVGPOOL *****\n";
        
        // Create input tensor (batch_size=1, channels=1, length=6)
        Tensor inputmp({1, 1, 6}, use_gpu);
        std::vector<float> inputmp_data = {1, 2, 3, 4, 5, 6};
        inputmp.load_data(inputmp_data);
        print_tensor(inputmp, "Original Tensor");

        // Perform average pooling with kernel=2, stride=2, no padding
        Tensor avg_pooled = inputmp.avgpool(2, 2, false);
        print_tensor(avg_pooled, "Average Pooled Tensor");

        // Expected results
        const std::vector<int> expected_shape = {1, 1, 3};
        const std::vector<float> expected_data = {1.5f, 3.5f, 5.5f};
        const float epsilon = 1e-5f;

        // 1. Check output shape
        if (avg_pooled.shape() != expected_shape) {
            std::cerr << "Shape mismatch! Expected [1, 1, 3], got [";
            for (size_t i = 0; i < avg_pooled.shape().size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << avg_pooled.shape()[i];
            }
            std::cerr << "]\n";
            return false;
        }

        // 2. Check output data
        std::vector<float> pooled_data = avg_pooled.get_data();
        bool data_ok = true;
        for (size_t i = 0; i < expected_data.size(); ++i) {
            if (std::abs(pooled_data[i] - expected_data[i]) > epsilon) {
                std::cerr << "Data mismatch at index " << i 
                          << ": expected " << expected_data[i]
                          << ", got " << pooled_data[i] << "\n";
                data_ok = false;
            }
        }
        if (!data_ok) return false;

        // 3. Verify original tensor is unchanged
        Tensor original_copy({1, 1, 6}, use_gpu);
        original_copy.load_data(inputmp_data);
        if (!compare_tensors(inputmp, original_copy, epsilon)) {
            std::cerr << "Original tensor was modified during avgpool operation!\n";
            return false;
        }

        std::cout << "Avgpool test OK\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in avgpool test: " << e.what() << "\n";
        return false;
    }
}

bool test_masked_assignment(bool use_gpu) {
    try {
        std::cout << "***** TEST MASKED ASSIGNMENT *****\n";
        const float epsilon = 1e-5f;

        // Create original tensor and mask
        Tensor the_tensor({2, 3}, use_gpu);
        Tensor the_mask({2, 3}, use_gpu);
        
        // Initialize data
        the_tensor.load_data({1, 2, 3, 4, 5, 6});
        the_mask.load_data({0, 1, 0, 1, 0, 1});
        
        // Preserve original mask by creating a copy manually
        Tensor original_mask(the_mask.shape(), use_gpu);
        original_mask.load_data(the_mask.get_data());
        
        // Perform operation
        the_tensor = {the_mask, 10.0f};
        
        // Create expected result
        Tensor expected({2, 3}, use_gpu);
        expected.load_data({1, 10, 3, 10, 5, 10});

        // 1. Verify result
        if (!compare_tensors(the_tensor, expected, epsilon)) {
            std::cerr << "Masked assignment result mismatch!\n";
            print_tensor(the_tensor, "Actual Result");
            print_tensor(expected, "Expected Result");
            return false;
        }

        // 2. Verify mask unchanged
        if (!compare_tensors(the_mask, original_mask, epsilon)) {
            std::cerr << "Mask tensor was modified during assignment!\n";
            return false;
        }

        std::cout << "Masked assignment test passed!\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in masked assignment test: " << e.what() << "\n";
        return false;
    }
}

// Test the accessor methods at and slice
bool test_at_method(bool use_gpu) {
    try {
        std::cout << "***** TEST AT() METHOD *****\n";
        Tensor t({2, 3, 4}, use_gpu);
        
        // Initialize data with pattern: i * ((i % 3 == 0) ? -1 : 1)
        std::vector<float> data(24);
        for (int i = 0; i < 24; ++i) {
            data[i] = i * ((i % 3 == 0) ? -1 : 1);
        }
        t.load_data(data);

        // For GPU tensors: Copy to CPU first
        Tensor t_cpu = t;
        if (use_gpu) {
            t_cpu = Tensor(t.shape(), false);
            t_cpu.load_data(t.get_data());
        }

        const float epsilon = 1e-5f;
        bool success = true;

        // Test valid indices (CPU only)
        if (!use_gpu) {
            std::vector<std::pair<std::vector<int>, float>> test_cases = {
                {{0, 0, 0}, 0},     // Index 0: 0 * -1 = 0
                {{1, 2, 3}, 23},    // Index 23: 23 * 1 = 23
                {{0, 1, 2}, -6},    // Index 6: 6 * -1 = -6
                {{1, 0, 3}, -15}    // Index 15: 15 * -1 = -15
            };

            for (const auto& [indices, expected] : test_cases) {
                float val = t_cpu.at(indices);
                if (std::abs(val - expected) > epsilon) {
                    std::cerr << "At " << format_indices(indices)
                              << " expected " << expected
                              << " got " << val << "\n";
                    success = false;
                }
            }
        }

        // Test invalid indices (CPU only)
        if (!use_gpu) {
            std::vector<std::vector<int>> bad_indices = {
                {2, 0, 0}, {1, 3, 0}, {0, 0, 4}, {0, 0}
            };
            for (const auto& indices : bad_indices) {
                try {
                    t_cpu.at(indices);
                    std::cerr << "Failed to throw for invalid indices: "
                              << format_indices(indices) << "\n";
                    success = false;
                } catch (const std::exception&) {
                    // Expected exception
                }
            }
        }

        // Test GPU tensor access throws
        if (use_gpu) {
            try {
                t.at({0, 0, 0});
                std::cerr << "Failed to throw for GPU tensor access\n";
                success = false;
            } catch (const std::exception& e) {
                // Expected exception
            }
        }

        if (success) {
            std::cout << "at() test passed!\n\n";
            return true;
        }
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in at() test: " << e.what() << "\n";
        return false;
    }
}

bool test_slice_method(bool use_gpu) {
    try {
        std::cout << "***** TEST SLICE() METHOD *****\n";
        const float epsilon = 1e-5f;

        // Create 4D tensor (2x3x4x2)
        Tensor original({2, 3, 4, 2}, use_gpu);
        std::vector<float> data(48);
        for (int i = 0; i < 48; ++i) data[i] = i;
        original.load_data(data);

        // Handle GPU by copying to CPU first
        Tensor original_cpu = original;
        if (use_gpu) {
            original_cpu = Tensor(original.shape(), false);
            original_cpu.load_data(original.get_data());
        }

        // Test 1: Complex slice
        auto sliced = original_cpu.slice({
            {1, 2},     // Keep index 1 in dim 0
            {0, 2},     // Keep indices 0,1 in dim 1
            {1, 4},     // Keep indices 1,2,3 in dim 2
            {0, 1}      // Keep index 0 in dim 3
        });

        // Verify new shape (1, 2, 3, 1)
        if (sliced.shape() != std::vector<int>{1, 2, 3, 1}) {
            std::cerr << "Shape mismatch. Actual: [";
            for (auto d : sliced.shape()) std::cerr << d << " ";
            std::cerr << "]\n";
            return false;
        }

        // Expected values (precomputed)
        std::vector<float> expected = {26.0f, 28.0f, 30.0f, 34.0f, 36.0f, 38.0f};
        Tensor expected_tensor({1, 2, 3, 1}, false);
        expected_tensor.load_data(expected);

        if (!compare_tensors(sliced, expected_tensor, epsilon)) {
            std::cerr << "Complex slice data mismatch!\n";
            print_tensor(sliced, "Actual");
            print_tensor(expected_tensor, "Expected");
            return false;
        }

        // Test 2: Single element slice (last element)
        auto single = original_cpu.slice({
            {1, 2},     // Index 1 in dim 0
            {2, 3},     // Index 2 in dim 1
            {3, 4},     // Index 3 in dim 2
            {1, 2}      // Index 1 in dim 3
        });

        Tensor single_expected({1, 1, 1, 1}, false);
        single_expected.load_data({47.0f}); // Correct value: 47

        if (!compare_tensors(single, single_expected, epsilon)) {
            std::cerr << "Single element slice failed!\n";
            print_tensor(single, "Actual");
            print_tensor(single_expected, "Expected");
            return false;
        }

        std::cout << "slice() test passed!\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in slice() test: " << e.what() << "\n";
        return false;
    }
}

// Test copy
bool test_copy(bool use_gpu) {
    try {
        std::cout << "***** TEST COPY *****\n";
        const float epsilon = 1e-5f;

        // Create original tensor
        Tensor original({2, 3}, use_gpu);
        std::vector<float> original_data = {1, 2, 3, 4, 5, 6};
        original.load_data(original_data);
        print_tensor(original, "Original Tensor");

        // Create copy
        Tensor copied = original.copy();
        print_tensor(copied, "Copied Tensor");

        // 1. Verify initial state
        if (!compare_tensors(original, copied, epsilon)) {
            std::cerr << "Initial copy mismatch!\n";
            return false;
        }

        // 2. Verify device consistency
        if (original.use_gpu() != copied.use_gpu()) {
            std::cerr << "Device mismatch! Original: " 
                      << (original.use_gpu() ? "GPU" : "CPU")
                      << " Copy: " 
                      << (copied.use_gpu() ? "GPU" : "CPU") << "\n";
            return false;
        }

        // 3. Modify original and verify copy remains unchanged
        std::vector<float> modified_data = {7, 8, 9, 10, 11, 12};
        original.load_data(modified_data);
        
        Tensor expected_unchanged(original.shape(), use_gpu);
        expected_unchanged.load_data(original_data);
        
        if (!compare_tensors(copied, expected_unchanged, epsilon)) {
            std::cerr << "Copy changed after original modification!\n";
            print_tensor(copied, "Current Copy");
            print_tensor(expected_unchanged, "Expected Copy");
            return false;
        }

        // 4. Modify copy and verify original stays modified
        std::vector<float> copy_modified_data = {13, 14, 15, 16, 17, 18};
        copied.load_data(copy_modified_data);
        
        Tensor expected_modified_original(original.shape(), use_gpu);
        expected_modified_original.load_data(modified_data);
        
        if (!compare_tensors(original, expected_modified_original, epsilon)) {
            std::cerr << "Original changed after copy modification!\n";
            print_tensor(original, "Current Original");
            print_tensor(expected_modified_original, "Expected Original");
            return false;
        }

        // 5. Verify shape preservation
        if (original.shape() != copied.shape()) {
            std::cerr << "Shape mismatch after modifications!\n";
            std::cerr << "Original shape: [";
            for (auto d : original.shape()) std::cerr << d << " ";
            std::cerr << "]\nCopy shape: [";
            for (auto d : copied.shape()) std::cerr << d << " ";
            std::cerr << "]\n";
            return false;
        }

        std::cout << "Copy test passed!\n\n";
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in copy test: " << e.what() << "\n";
        return false;
    }
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
    if (!test_multiply(use_gpu)) {
        std::cerr << "ERROR test_multiply failed\n";
        return false;
    }

    // Test element-wise division: G = A / B
    if (!test_divide(use_gpu)) {
        std::cerr << "ERROR test_divide failed\n";
        return false;
    }

    // Test scalar multiplication: D = alpha * A
    if (!test_multiply_scalar(use_gpu)) {
        std::cerr << "ERROR test_multiply_scalar failed\n";
        return false;
    }

    // Test power operation
    if (!test_power(use_gpu)) {
        std::cerr << "ERROR test_power failed\n";
        return false;
    }

    // Test 1D convolution
    if (!test_conv1d(use_gpu)) {
        std::cerr << "ERROR test_connv1d failed\n";
        return false;
    }

    // Test 2D convolution
    if (!test_conv2d(use_gpu)) {
        std::cerr << "ERROR test_conv2d failed\n";
        return false;
    }

    // Test 3D convolution
    if (!test_conv3d(use_gpu)) {
        std::cerr << "ERROR test_conv3d failed\n";
        return false;
    }

    // Test power
    if (!test_power(use_gpu)) {
        std::cerr << "ERROR test_power failed\n";
        return false;
    }

    // Test sum along axis 1
    if (!test_sum_along_axis(use_gpu)) {
        std::cerr << "ERROR test_sum_along_axis failed\n";
        return false;
    }


    // Test mean along axis
    if (!test_mean_along_axis(use_gpu)) {
        std::cerr << "ERROR test_mean_along_axis failed\n";
        return false;
    }

    // Test max (scalar)
    if (!test_max(use_gpu)) {
        std::cerr << "ERROR test_max failed\n";
        return false;
    }

    // Test max along axis
    if (!test_max_along_axis(use_gpu)) {
        std::cerr << "ERROR test_max_along_axis failed\n";
        return false;
    }

    // Test min
    if (!test_min(use_gpu)) {
        std::cerr << "ERROR test_min failed\n";
        return false;
    }

    // Test min_along_axis
    if (!test_min_along_axis(use_gpu)) {
        std::cerr << "ERROR test_min_along_axis failed\n";
        return false;
    }

    // Test argmax
    if (!test_argmax(use_gpu)) {
        std::cerr << "ERROR test_argmax failed\n";
        return false;
    }

    // test argmin
    if (!test_argmin(use_gpu)) {
        std::cerr << "ERROR test_argmin failed\n";
        return false;
    }

    // Test matmul
    if (!test_matmul(use_gpu)) {
        std::cerr << "ERROR test_matmul failed\n";
        return false;
    }

    if (!test_inv(use_gpu)) {
        std::cerr << "ERROR test_inv failed\n";
        return false;
    }

    // Test transpose
    if (!test_transpose(use_gpu)) {
        std::cerr << "ERROR test_transpose failed\n";
        return false;
    }

    // Test determinant
    if (!test_determinant(use_gpu)) {
        std::cerr << "ERROR test_determinant failed\n";
        return false;
    }

    // Test eigen
    if (!test_eigen(use_gpu)) {
        std::cerr << "ERROR test_eigen failed\n";
        return false;
    }

    // Test dot product using einsum
    if (!test_dot_product_einsum(use_gpu)) {
        std::cerr << "ERROR test_dot_product_einsum failed\n";
        return false;
    }

    // Test reshape
    if (!test_reshape(use_gpu)) {
        std::cerr << "Error test_reshape failed\n";
        return false;
    }

    // Test flatten
    if (!test_flatten(use_gpu)) {
        std::cerr << "Error test_flatten failed\n";
        return false;
    }

    // Test expand_dims
    if (!test_expand_dims(use_gpu)) {
        std::cerr << "ERROR test_expand_dims failed\n";
        return false;
    }

    if (!test_squeeze(use_gpu)) {
        std::cerr << "ERROR test_squeeze failed\n";
        return false;
    }

    // Test concat
    if (!test_concat(use_gpu)) {
        std::cerr << "ERROR test_concat failed\n";
        return false;
    }

    if (!test_stack(use_gpu)) {
        std::cerr << "ERROR test_stack failed\n";
        return false;
    }

    // Test permute
    if (!test_permute(use_gpu)) {
        std::cerr << "ERROR test_permute failed\n";
        return false;
    }

    if (!test_edge_squeeze_expand(use_gpu)) {
        std::cerr << "ERROR test_edge_squeeze_expand failed\n";
        return false;
    }

    // Test broadcast
    if (!test_broadcast(use_gpu)) {
        std::cerr << "ERROR test_broadcast failed\n";
        return false;
    }

    // Test comparisons
    if (!test_comparisons(use_gpu)) {
        std::cerr << "ERROR test_comparisons failed\n";
        return false;
    }

    // Test masked comparisons
    if (!test_masked_comparisons(use_gpu)) {
        std::cerr << "ERROR test_masked_comparisons failed\n";
        return false;
    }

    // Test maxpool
    if (!test_maxpool(use_gpu)) {
        std::cerr << "ERROR test_maxpool failed\n";
        return false;
    }

    // Test avgpool
    if (!test_avgpool(use_gpu)) {
        std::cerr << "ERROR test_avgpool failed\n";
        return false;
    }

    // TESTING REPEAT
    if (!test_repeat(use_gpu)) {
        std::cerr << "ERROR test_repeat failed\n";
        return false;
    }

    // Test masked assignment
    if (!test_masked_assignment(use_gpu)) {
        std::cerr << "ERROR test_masked_assignment failed\n";
        return false;
    }

    // Test copy
    if (!test_copy(use_gpu)) {
        std::cerr << "ERROR test_copy failed\n";
        return false;
    }
    
    // Test at method
    if (!test_at_method(use_gpu)) {
        std::cerr << "ERROR test_at_method failed\n";
        return false;
    }

    // Test slice method
    if (!test_slice_method(use_gpu)) {
        std::cerr << "ERROR test_slice_method failed\n";
        return false;
    }

    // Test copy method
    if (!test_copy(use_gpu)) {
        std::cerr << "ERROR test_copy failed\n";
        return false;
    }

    return true;
}