cmake_minimum_required(VERSION 3.4...3.18)

# Work around a version compatibility issue between nvcc and g++ when compiling
# the cub library and its dependency on std::functional.
# For me, nvcc 11.5 w/ g++ 11.3 doesn't work, but nvcc 11.5 w/ g++ 10.4 does.
# The -lineinfo flag ensures line number debug information is available when
# using tools like compute-sanitizer.
set(CMAKE_CUDA_FLAGS "-ccbin=g++-10 -lineinfo")

# Uncomment to print out each command as it's executed:
# set(CMAKE_VERBOSE_MAKEFILE ON)

project(kernel LANGUAGES CXX CUDA)
project(c_demo LANGUAGES CXX CUDA)

# The Python module version of this code.
find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
pybind11_add_module(
    kernel
    development.cu
    fitness.cu
    gol_simulation.cu
    python_module.cc
    reproduction.cu
    selection.cu
    simulator.cu)

# For debugging, build a pure C++ simulation demo that doesn't include Pybind11
add_executable(
    cpp_demo
    development.cu
    fitness.cu
    gol_simulation.cu
    reproduction.cu
    selection.cu
    simulator.cu)

# Force all C++ code to build with g++ 10, so that the host-side bytecode is
# linkcable with the device-side bytecode. It seems like this has to come after
# loading the pybind11 libraries to effect them.
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

# Necessary to include __device__ functions across file boundaries.
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cpp_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Uncomment to generate PTX files to examine register allocation.
# add_library(simulator OBJECT simulator.cu)
# set_property(TARGET simulator PROPERTY CUDA_PTX_COMPILATION ON)
