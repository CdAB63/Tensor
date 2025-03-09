# TensorLibrary

TensorLibrary is a C++ tensor computation library with optional CUDA support. It provides efficient tensor operations for numerical computing and deep learning applications. The library automatically falls back to CPU operations when CUDA is not available.

## Features
- Supports basic tensor operations: addition, subtraction, multiplication, division.
- Dot product and matrix multiplication.
- Element-wise operations (power, comparison, etc.).
- Convolution (1D, 2D, 3D) and pooling (max, average).
- CUDA acceleration with CPU fallback.
- Designed for easy integration with other projects.

## Installation & Building

### Prerequisites
Ensure you have the following installed:
- C++ compiler (GCC, Clang, MSVC, etc.)
- CMake (>=3.10)
- CUDA Toolkit (if enabling CUDA support)

### Clone Repository
```sh
git clone https://github.com/CdAB63/Tensor.git
cd Tensor
```

### Build Instructions

#### Building without CUDA
```sh
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF
make
```

#### Building with CUDA
```sh
mkdir build && cd build
cmake .. -DUSE_CUDA=ON
make
```

### Running Tests
```sh
./build/TensorTest
```

## API Documentation

### `Tensor` Class
The `Tensor` class provides an abstraction for multi-dimensional arrays with support for arithmetic operations and matrix computations.

#### Constructor
```cpp
Tensor(std::vector<int> shape, bool use_cuda = false);
```
Creates a tensor with the given shape. If `use_cuda` is true, operations are performed on the GPU.

**Example:**
```cpp
Tensor t({2, 3}); // Creates a 2x3 tensor on CPU
```

#### Addition
```cpp
Tensor operator+(const Tensor& other) const;
```
Returns a new tensor with element-wise addition.

**Example:**
```cpp
Tensor a({2, 2});
Tensor b({2, 2});
Tensor c = a + b;
```

#### Subtraction
```cpp
Tensor operator-(const Tensor& other) const;
```
Returns a new tensor with element-wise subtraction.

#### Multiplication
```cpp
Tensor operator*(const Tensor& other) const;
```
Returns a new tensor with element-wise multiplication.

#### Division
```cpp
Tensor operator/(const Tensor& other) const;
```
Returns a new tensor with element-wise division.

#### Dot Product
```cpp
Tensor dot(const Tensor& other) const;
```
Computes the dot product of two tensors.

**Example:**
```cpp
Tensor a({3, 3});
Tensor b({3, 3});
Tensor result = a.dot(b);
```

#### Matrix Multiplication
```cpp
Tensor matmul(const Tensor& other) const;
```
Performs matrix multiplication.

#### Convolution (1D, 2D, 3D)
```cpp
Tensor conv1d(const Tensor& kernel);
Tensor conv2d(const Tensor& kernel);
Tensor conv3d(const Tensor& kernel);
```
Applies convolution operations.

#### Pooling Operations
```cpp
Tensor maxpool2d(int kernel_size, int stride);
Tensor avgpool2d(int kernel_size, int stride);
```
Performs max and average pooling.

## License
TensorLibrary is licensed under the Apache 2.0 License.

