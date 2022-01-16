cmake_minimum_required (VERSION 3.9)

project(raycasting_cmake LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

# Modify if you need a different default value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
endif()

enable_language(CUDA)

# List all your code files here
add_library(raycasting_cmake SHARED
  raycasting.cpp
  weighted_resampling.cu
)
target_compile_features(raycasting_cmake PRIVATE cxx_std_11)
target_link_libraries(raycasting_cmake PRIVATE ${TORCH_LIBRARIES} Python::Python)

# Use if the default GCC version gives issues.
# Similar syntax is used if we need better compilation flags.
target_compile_options(raycasting_cmake PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ccbin g++-9>)

# Use a variant of this if you're on an earlier cmake than 3.18
# target_compile_options(pytorch_cmake_example PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)