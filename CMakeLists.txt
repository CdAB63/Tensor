cmake_minimum_required(VERSION 3.10)
project(TensorLibrary)

# Set policy to suppress CMP0146 warning
cmake_policy(SET CMP0146 NEW)

# Enable CUDA if available
option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)  # Modern CUDA detection
    enable_language(CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
endif()

# Include headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Eigen3 libraries required
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(SOURCES
    src/Tensor.cpp
)

if(USE_CUDA)
    list(APPEND SOURCES src/cuda_kernels.cu)
endif()

# Add debug flags for CUDA and C++
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)  # CUDA debug flag
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-g>)   # C++ debug flag
endif()

# Create the shared library
add_library(TensorLibrary SHARED ${SOURCES})

# Link CUDA libraries if enabled
if(USE_CUDA)
    target_link_libraries(TensorLibrary PRIVATE CUDA::cusolver CUDA::cublas)
endif()

# Add an executable for testing
add_executable(TensorTest src/main.cpp)
target_link_libraries(TensorTest PRIVATE TensorLibrary)
