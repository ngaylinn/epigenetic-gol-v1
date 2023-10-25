/*
 * Compute fitness for GOL simulations. This module consists of the
 * FitnessObserver template class and implementations of that class
 * for each FitnessGoal in the project.
 */

#ifndef __FITNESS_H__
#define __FITNESS_H__

#include <cstdint>

#include "environment.h"

namespace epigenetic_gol_kernel {

/*
 * A class for computing the fitness score of a GOL simulation by observing
 * each frame as it is rendered. Rather than holding the whole simulation in
 * GPU memory (which is 64x64x100 bytes = 400kb), this object stores 8 bytes
 * of incremental fitness data for each Cell in the GOL board (32kb), which
 * should fit in registers rather than global memory.
 */
template<FitnessGoal GOAL>
class FitnessObserver {
    private:
        // Scratch space for storing partial fitness data
        uint32_t scratch_a[CELLS_PER_THREAD] = {};
        uint32_t scratch_b[CELLS_PER_THREAD] = {};

        // Update is a goal-specific implementation of observe that updates
        // the values in scratch_a and scratch_b to capture any relevant
        // fitness data from the given time step and range of Cells.
        // The member variables scratch_a and scratch_b are
        // passed as arguments so that goal-specific implementations can rename
        // them to reflect their goal-specific meanings. This
        // version of update can observe the whole frame for this step
        // (in global memory, which is relatively expensive) to identify
        // global patterns within the board.
        __device__ void update(
                const int& step, const int& row, const int& col,
                const Frame& frame, uint32_t& scratch_a, uint32_t& scratch_b);

        // Same as above, except this version of update can only observe the
        // region of the GOL board being evaluated (which may be in registers,
        // the fastest GPU memory). This is preferred when global visibiliy
        // isn't needed.
        __device__ void update(
                const int& step, const int& row, const int& col,
                const Cell& cell, uint32_t& scratch_a, uint32_t& scratch_b);

        // Combine the goal-specific partial fitness data into a single fitness
        // score for an entire simulation.
        __device__ void finalize(
                const uint32_t& sum_a, const uint32_t& sum_b, Fitness* result);

    public:
        // Observe the GOL game board at the given time step and capture relevant
        // data for computing overall fitness of the simulation. Each call will
        // consider CELLS_PER_THREAD cells in a row, starting from position
        // (row, col). This function is passed both a local and global view of
        // the GOL board, but only one will be used. This allows working in
        // registers when possible, which gives a significant performance boost.
        __device__ void observe(
                const int& step, const int& row, const int& col,
                const Cell local[CELLS_PER_THREAD], const Frame& global);

        // After observe has been run once for every step and all Cells on the
        // board, call this function to reduce all the partial fitness data
        // captured into a single score for the entire simulation.
        __device__ void reduce(Fitness* result);
};

void compute_entropy(
        const int population_size, Video* videos, Fitness* fitness);

} // namespace epigenetic_gol_kernel

#endif
