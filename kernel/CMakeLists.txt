cmake_minimum_required(VERSION 3.4...3.18)

# Save compiler option details, which a language server can use for improved
# diagnostics when developing this project.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Work around a version compatibility issue between nvcc and g++ when compiling
# the cub library and its dependency on std::functional.
# For me, nvcc 11.5 w/ g++ 11.3 doesn't work, but nvcc 11.5 w/ g++ 10.4 does.
# The -lineinfo flag ensures line number debug information is available when
# using tools like compute-sanitizer.
set(CMAKE_CUDA_FLAGS "-ccbin=g++-10 -lineinfo")

# Uncomment to print out each command as it's executed for debugging purposes.
# set(CMAKE_VERBOSE_MAKEFILE ON)

project(kernel LANGUAGES CXX CUDA)
project(c_demo LANGUAGES CXX CUDA)

# Bundle all the C++ code for this project as a Python module.
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
# or Python in any way.
add_executable(
    cpp_demo
    development.cu
    fitness.cu
    gol_simulation.cu
    reproduction.cu
    selection.cu
    simulator.cu)

# The nvcomp library is needed for the ENTROPY fitness goal. This dependency
# can be removed if you also disable the ENTROPY fitness code.
find_package(nvcomp 3.0.2 REQUIRED)
target_link_libraries(
    kernel PRIVATE nvcomp::nvcomp)
target_link_libraries(
    cpp_demo PRIVATE nvcomp::nvcomp)

# Force all C++ code to build with g++ 10, so that code that runs on host and
# device is linkable with host-only code. It seems like this has to come after
# loading the pybind11 libraries to affect them.
set(CMAKE_CXX_COMPILER /usr/bin/g++-10)

# Configuration options required to reference __device__ functions across file
# boundaries.
set_target_properties(kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cpp_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Uncomment to generate PTX files to examine register allocation.
# add_library(simulator OBJECT simulator.cu)
# set_property(TARGET simulator PROPERTY CUDA_PTX_COMPILATION ON)
