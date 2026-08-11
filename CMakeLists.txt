cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchattacks_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Locate LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directory
include_directories(include ${TORCH_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/attack.cpp
    src/fgsm.cpp
    src/pgd.cpp
    src/kernels/pgd_kernel.cu
)

# Build static library
add_library(torchattacks_lib STATIC ${SOURCES})
target_link_libraries(torchattacks_lib PUBLIC ${TORCH_LIBRARIES})

# Set CUDA properties
set_target_properties(torchattacks_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89;90"
)

# Build benchmark executable
add_executable(benchmark examples/benchmark.cpp)
target_link_libraries(benchmark PRIVATE torchattacks_lib ${TORCH_LIBRARIES})