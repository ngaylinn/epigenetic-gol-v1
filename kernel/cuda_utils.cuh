#ifndef __CUDA_UTILS__H_
#define __CUDA_UTILS__H_

#include <stdio.h>

namespace epigenetic_gol_kernel {

#define CUDA_CALL(val) check((val), #val, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() checkLast(__FILE__, __LINE__)

namespace {

void check(cudaError_t error, const char* const src_string,
        const char* const file, const int line) {
    if (error == cudaSuccess) return;
    printf("CUDA Error @ %s:%d\n", file, line);
    printf("    %s: %s\n", src_string, cudaGetErrorString(error));
}

void checkLast(const char* const file, const int line) {
    check(cudaGetLastError(), "Last error", file, line);
}

unsigned int max_threads() {
    cudaDeviceProp properties;
    // This project is currently designed to run on a single GPU.
    CUDA_CALL(cudaGetDeviceProperties(&properties, 0));
    return properties.maxThreadsPerBlock;
}

} // namespace

const unsigned int MAX_THREADS = max_threads();

} // namespace epigenetic_gol_kernel

#endif
