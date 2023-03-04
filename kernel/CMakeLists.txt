cmake_minimum_required(VERSION 3.4...3.18)

# Work around a version compatibility issue between nvcc and g++ when compiling
# the cub library and its dependency on std::functional.
# For me, nvcc 11.5 w/ g++ 11.3 doesn't work, but nvcc 11.5 w/ g++ 10.4 does.
set(CMAKE_CUDA_FLAGS "-ccbin=g++-10")

# Uncomment to print out each command as it's executed:
# set(CMAKE_VERBOSE_MAKEFILE ON)

project(kernel LANGUAGES CXX CUDA)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
pybind11_add_module(
    kernel
    development.cu
    fitness.cu
    genotype.cu
    gol_simulation.cu
    interpreter.cu
    python_module.cc
    selection.cu
    simulator.cu)

# Force all C++ code to build with g++ 10, so that the host-side bytecode is
# linkcable with the device-side bytecode. It seems like this has to come after
# loading the pybind11 libraries to effect them.
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

# Necessary to include __device__ functions across file boundaries.
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Uncomment to generate PTX files to examine register allocation.
# add_library(simulator OBJECT simulator.cu)
# set_property(TARGET simulator PROPERTY CUDA_PTX_COMPILATION ON)
