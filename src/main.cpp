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
    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);
    
    // Test argmin along axis 1
    Tensor argmin_result = A1.argmin(1);
    print_tensor(argmin_result, "Argmin along axis 1");
    
    Tensor A2({2, 2}, use_gpu);
    A2.load_data({4.0f, 1.0f, 2.0f, 3.0f});
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

    // Create a tensor (2x3x4)
    Tensor A1({2, 3, 4}, use_gpu);
    std::vector<float> A1_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; ++i) A1_data[i] = static_cast<float>(i);
    A1.load_data(A1_data);

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