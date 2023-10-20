/*
 * Utilities for working with CUDA.
 *
 * The purpose of this header is mostly to avoid verbose boilerplate and common
 * definitions from getting repeated across all the files of this project.
 */

#ifndef __CUDA_UTILS__H_
#define __CUDA_UTILS__H_

#include <stdio.h>
// TODO: Unused?
#include <vector>
// TODO: Consider upgrading to C++20 so you don't need the experimetnal version
#include <experimental/source_location>

#include <curand_kernel.h>

namespace epigenetic_gol_kernel {

// CUDA error-checking macros. It's recommended to use these after every CUDA
// operation so that asynchronous errors are noticed sooner rather than later.
// CUDA_CALL wraps a CUDA function call, while CUDA_CHECK_ERROR is called
// immediately after a kernel launch.
#define CUDA_CALL(val) check((val), #val, __FILE__, __LINE__)
// TODO: Is CUDA_CHECK_ERROR still working? Didn't catch error in GOL kernel
// invocation.
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

// The maximum number of threads per block on this device.
const unsigned int MAX_THREADS = max_threads();


/*
 * A convenience class for managing device-side memory allocations.
 *
 * This class is used to create single objects or arrays of objects on the
 * GPU device, handling all the CUDA memory management behind the scenes.
 */
template<typename T>
class DeviceData {
    protected:
        T* data;
        const int size;

    public:
        DeviceData(
                const int size=1, // Array size, or 1 for single object.
                // For line numbers in error reporting. Don't pass an argument.
                const std::experimental::source_location location =
                    std::experimental::source_location::current()
                ) : size(size) {
            check(cudaMalloc(&data, sizeof(T) * size),
                    location.function_name(), location.file_name(),
                    location.line());
        }

        DeviceData(
                const int size,
                const T* h_data, // initialize to host-side data
                const std::experimental::source_location location =
                    std::experimental::source_location::current()
                ) : DeviceData(size) {
            copy_from_host(h_data, location);
        }

        // TODO: If you give size a default value above, do you need a
        // separate definition for this function?
        DeviceData(const T* h_data) : DeviceData(1, h_data) {}

        ~DeviceData() {
            CUDA_CALL(cudaFree(data));
        }

        void swap(DeviceData<T>& other) {
            T* temp = this->data;
            this->data = other.data;
            other.data = temp;
        }

        void copy_from_host(
                const T* h_data,
                const std::experimental::source_location location =
                    std::experimental::source_location::current()) {
            check(cudaMemcpy(data, h_data, sizeof(T) * size,
                        cudaMemcpyHostToDevice),
                    location.function_name(), location.file_name(),
                    location.line());
        }

        void copy_to_host(
                T* h_data,
                const std::experimental::source_location location =
                    std::experimental::source_location::current()) const {
            check(cudaMemcpy(h_data, data, sizeof(T) * size,
                        cudaMemcpyDeviceToHost),
                    location.function_name(), location.file_name(),
                    location.line());
        }

        T* copy_to_host(
                const std::experimental::source_location location =
                    std::experimental::source_location::current()) const {
            T* result = new T[size];
            copy_to_host(result, location);
            return result;
        }

        operator const T*() const {
            return data;
        }

        operator T*() {
            return data;
        }

        // TODO: Would it be possible / better to remove this and use
        // a * operator directly on a DeviceData object?
        operator const T&() const {
            return *data;
        }
};

namespace {

__global__ void InitRngsKernel(
        curandState* rngs, int size, unsigned int seed) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;

    curand_init(seed, index, 0, &rngs[index]);
}

} // namespace

// Initialize an array of curandState objects on the GPU device.
inline void seed_rngs(
        curandState* rngs, unsigned int size, unsigned int seed_value) {
    InitRngsKernel<<<
        (size + MAX_THREADS - 1) / MAX_THREADS,
        min(size, MAX_THREADS)
    >>>(rngs, size, seed_value);
}

} // namespace epigenetic_gol_kernel

#endif
