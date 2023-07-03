#ifndef __FITNESS_H__
#define __FITNESS_H__

#include <cstdint>

#include "environment.h"

namespace epigenetic_gol_kernel {

template<FitnessGoal GOAL>
class FitnessObserver {
    private:
        uint32_t scratch_a[CELLS_PER_THREAD] = {};
        uint32_t scratch_b[CELLS_PER_THREAD] = {};
        __device__ void update(
                const int& step, const int& row, const int& col,
                const Frame& frame, uint32_t& scratch_a, uint32_t& scratch_b);
        __device__ void update(
                const int& step, const int& row, const int& col,
                const Cell& cell, uint32_t& scratch_a, uint32_t& scratch_b);
        __device__ void finalize(
                const uint32_t& sum_a, const uint32_t& sum_b, Fitness* result);

    public:
        __device__ void observe(
                const int& step, const int& row, const int& col,
                const Cell local[CELLS_PER_THREAD], const Frame& global);
        __device__ void reduce(Fitness* result);
};

} // namespace epigenetic_gol_kernel

#endif
