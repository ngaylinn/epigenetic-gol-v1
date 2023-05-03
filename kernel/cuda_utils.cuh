#ifndef __CUDA_UTILS__H_
#define __CUDA_UTILS__H_

#include <stdio.h>
#include <vector>

#include <curand_kernel.h>

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
        DeviceData(int size=1) : size(size) {
            CUDA_CALL(cudaMalloc(&data, sizeof(T) * size));
        }

        DeviceData(int size, const T* h_data) : DeviceData(size) {
            copy_from_host(h_data);
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

        void copy_from_host(const T* h_data) {
            CUDA_CALL(cudaMemcpy(data, h_data, sizeof(T) * size,
                        cudaMemcpyHostToDevice));
        }

        void copy_to_host(T* h_data) const {
            CUDA_CALL(cudaMemcpy(h_data, data, sizeof(T) * size,
                        cudaMemcpyDeviceToHost));
        }

        T* copy_to_host() const {
            T* result = new T[size];
            copy_to_host(result);
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

constexpr unsigned int SEED = 42;

__global__ void InitRngsKernel(int population_size, curandState* rngs) {
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index >= population_size) return;

    curand_init(SEED, population_index, 0, &rngs[population_index]);
}

} // namespace

/*
 * A convenience class for managing curandState objects.
 *
 * This class handles allocation, initialization, serialization, and
 * deserialization of curandState arrays of arbitrary size.
 */
class CurandStates {
    protected:
        int size;
        DeviceData<curandState> data;

    public:
        CurandStates(int size)
            : size(size), data(size) {
            reset();
        }

        void reset() {
            InitRngsKernel<<<
                (size + MAX_THREADS - 1) / MAX_THREADS,
                min(size, MAX_THREADS)
            >>>(size, data);
        }

        const std::vector<unsigned char> get_state() const {
            int state_size = sizeof(curandState) * size;
            std::vector<unsigned char> result(state_size);
            CUDA_CALL(cudaMemcpy(result.data(), data, state_size,
                        cudaMemcpyDeviceToHost));
            return result;
        }

        void restore_state(std::vector<unsigned char> state) {
            if (state.size() != size) {
                perror("State object has incorrect size; cannot restore.\n");
            }
            CUDA_CALL(cudaMemcpy(data, state.data(),
                        sizeof(curandState) * size, cudaMemcpyHostToDevice));
        }

        operator curandState*() {
            return data;
        }

};

} // namespace epigenetic_gol_kernel

#endif
