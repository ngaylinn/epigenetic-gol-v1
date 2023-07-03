#ifndef __CUDA_UTILS__H_
#define __CUDA_UTILS__H_

#include <stdio.h>
#include <vector>
// TODO: Consider upgrading to C++20 so you don't need the experimetnal version
#include <experimental/source_location>

#include <curand_kernel.h>

namespace epigenetic_gol_kernel {

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

const unsigned int MAX_THREADS = max_threads();


/*
 * A convenience class for managing device-side memory allocations.
 *
 * This class can be used to create single objects or arrays of objects on the
 * GPU device, handling all the CUDA memory management behind the scenes.
 */
template<typename T>
class DeviceData {
    protected:
        T* data;
        const int size;

    public:
        DeviceData(
                int size=1,
                const std::experimental::source_location location =
                    std::experimental::source_location::current()
                ) : size(size) {
            check(cudaMalloc(&data, sizeof(T) * size),
                    location.function_name(), location.file_name(),
                    location.line());
        }

        DeviceData(
                int size,
                const T* h_data,
                const std::experimental::source_location location =
                    std::experimental::source_location::current()
                ) : DeviceData(size) {
            copy_from_host(h_data, location);
        }

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

inline void seed_rngs(
        curandState* rngs, unsigned int size, unsigned int seed_value) {
    InitRngsKernel<<<
        (size + MAX_THREADS - 1) / MAX_THREADS,
        min(size, MAX_THREADS)
    >>>(rngs, size, seed_value);
}

} // namespace epigenetic_gol_kernel

#endif
