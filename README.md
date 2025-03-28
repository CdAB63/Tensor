# Tensor Library (Complete Documentation)

[Content identical to previous response...]

<!-- FULL CONTENT FROM PREVIOUS RESPONSE BELOW -->

A GPU/CPU hybrid tensor computation library with full support for linear algebra, convolutions, pooling, broadcasting, and tensor manipulations. Designed for both machine learning and numerical computing.

## Table of Contents
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Method Reference](#method-reference)
  - [Tensor Creation & Basics](#tensor-creation--basics)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Linear Algebra](#linear-algebra)
  - [Convolution Operations](#convolution-operations)
  - [Pooling Operations](#pooling-operations)
  - [Reduction Operations](#reduction-operations)
  - [Tensor Manipulation](#tensor-manipulation)
  - [Advanced Operations](#advanced-operations)
- [Examples](#examples)
- [Compilation](#compilation)
- [License](#license)

## Installation
```bash
git clone https://github.com/your-repo/tensor-library.git
cd tensor-library
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug  # or Release
make -j4
