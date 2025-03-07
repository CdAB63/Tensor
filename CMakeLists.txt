cmake_minimum_required(VERSION 3.10)
project(TensorLibrary)

# Enable CUDA if available
option(USE_CUDA "Enable CUDA support" ON)

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
endif()

# Include headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/Tensor.cpp
)

if(USE_CUDA)
    list(APPEND SOURCES src/cuda_kernels.cu)
endif()

# Create the library
add_library(TensorLibrary ${SOURCES})

# Link CUDA libraries if enabled
if(USE_CUDA)
    target_link_libraries(TensorLibrary ${CUDA_LIBRARIES})
endif()

# Add an executable for testing
add_executable(TensorTest src/main.cpp)
target_link_libraries(TensorTest TensorLibrary)
