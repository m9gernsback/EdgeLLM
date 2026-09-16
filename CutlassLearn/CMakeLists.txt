cmake_minimum_required(VERSION 3.24)
project(CutlassLearn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5080 = Blackwell consumer, sm_120
set(CMAKE_CUDA_ARCHITECTURES 120)

# Keep line info so Nsight Compute can map SASS back to source
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(CudaBasics)
