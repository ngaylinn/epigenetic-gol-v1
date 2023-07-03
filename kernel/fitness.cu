#include "fitness.cuh"

#include <cstdint>

#include <cub/cub.cuh>

#include "environment.h"

namespace epigenetic_gol_kernel {

// ---------------------------------------------------------------------------
// FitnessObserver Implementation.
// ---------------------------------------------------------------------------

template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::observe(
        const int& step, const int& row, const int& col,
        const Cell local[CELLS_PER_THREAD], const Frame& global) {
    for (int i = 0; i < CELLS_PER_THREAD; i++) {
        // Only some fitness goals require a global view of the GOL simulation,
        // which is a bit less efficient than using a local view. The compiler
        // can optimize away the switch statement and directly call the correct
        // method for each goal.
        switch(GOAL) {
            case FitnessGoal::STILL_LIFE:
            case FitnessGoal::EXPLODE:
            case FitnessGoal::LEFT_TO_RIGHT:
            case FitnessGoal::THREE_CYCLE:
            case FitnessGoal::TWO_CYCLE:
                update(step, row, col+i, local[i], scratch_a[i], scratch_b[i]);

            case FitnessGoal::GLIDERS:
            case FitnessGoal::SYMMETRY:
            default:
                update(step, row, col+i, global, scratch_a[i], scratch_b[i]);
        }
    }
}

template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::reduce(Fitness* result) {
    auto reduce = cub::BlockReduce<uint32_t, THREADS_PER_BLOCK>();
    uint32_t sum_a = reduce.Sum(scratch_a);
    // Needed because both calls to Sum share the same workspace memory.
    __syncthreads();
    uint32_t sum_b = reduce.Sum(scratch_b);

    // Save the final result to global memory to return to the host.
    if (threadIdx.x == 0) {
        return finalize(sum_a, sum_b, result);
    }
}

// There's a version of FitnessObserver for every FitnessGoal, and each one
// must implement one of these two functions. Empty implementations are
// provided so the compiler won't complain about a missing definition for
// whichever version goes unused.
template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::update(
        const int& step, const int& row, const int& col, const Frame& frame,
        uint32_t& scratch_a, uint32_t& scratch_b) {}
template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& scratch_a, uint32_t& scratch_b) {}

// ---------------------------------------------------------------------------
// Explode
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::EXPLODE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& alive_on_first, uint32_t& alive_on_last) {
    if (step > 0 && step < NUM_STEPS - 1) return;

    if (step == 0) {
        alive_on_first = (cell == Cell::ALIVE);
    } else { // step == NUM_STEPS - 1
        alive_on_last = (cell == Cell::ALIVE);
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::EXPLODE>::finalize(
        const uint32_t& alive_on_first, const uint32_t& alive_on_last,
        Fitness* result) {
    *result = (100 * alive_on_last) / (1 + alive_on_first);
}


// ---------------------------------------------------------------------------
// Gliders
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::GLIDERS>::update(
        const int& step, const int& row, const int& col,
        const Frame& frame, uint32_t& history, uint32_t& repeating) {
    constexpr int row_delta = 1;
    constexpr int col_delta = 1;
    constexpr int time_delta = 4;
    constexpr int mask = 0b1 << time_delta;

    if (step < NUM_STEPS - time_delta - 1) return;

    const Cell& cell = frame[row][col];

    history = history << 1 | (cell == Cell::ALIVE);
    if (step < NUM_STEPS - 1) return;

    const bool last_cycle = history & mask;
    const bool this_cycle =
        (row + row_delta < WORLD_SIZE) &&
        (col + col_delta < WORLD_SIZE) &&
        (frame[row + row_delta][col + col_delta] == Cell::ALIVE);
    const bool is_static = (history & 0b1111) == 0b1111;
    repeating = last_cycle && this_cycle && !is_static;

    // TODO: Punishing the organism for live cells at the end helps, but can
    // provide a weak signal. Only punishing live cells that aren't repeating
    // can be better, but encourages lame solutions. Perhaps there's a better
    // way to do this?
    // Forget all the history except for the cell value on the last step,
    // so we can count up how many cells in total are alive.
    history = history & 0b1;
}

template<>
__device__ void FitnessObserver<FitnessGoal::GLIDERS>::finalize(
        const uint32_t& live_cells, const uint32_t& repeating,
        Fitness* result) {
    *result = (100 * repeating) / (1 + live_cells);
}


// ---------------------------------------------------------------------------
// LeftToRight
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& on_target_count, uint32_t& off_target_count) {
    const bool alive = (cell == Cell::ALIVE);

    // TODO: Commit to one implementation of this goal!

    //if (step == 0) {
    //    off_target_count = alive && col >= WORLD_SIZE / 2;
    //} else if (step == NUM_STEPS - 1) {
    //    on_target_count += alive && col >= WORLD_SIZE / 2;
    //    off_target_count += alive && col < WORLD_SIZE / 2;
    //}
    //return;

    // This version is equivalent to the prototype version of this goal.
    if (step == 0) {
        on_target_count = alive && col < WORLD_SIZE / 2;
        off_target_count = alive && col >= WORLD_SIZE / 2;
    } else if (step == NUM_STEPS - 1) {
        on_target_count += alive && col >= WORLD_SIZE / 2;
        off_target_count += alive && col < WORLD_SIZE / 2;
    }
    return;

    //constexpr uint32_t window = WORLD_SIZE / 2;
    //const int target =
    //    window + (step * (WORLD_SIZE - window - 1)) / (NUM_STEPS - 1);
    //const bool on_target = target >= col && target - col <= window;
    //on_target_count += alive && on_target;
    //off_target_count += alive && !on_target;

    // constexpr uint32_t window = WORLD_SIZE / 4;
    // const int target =
    //     window + (step * (WORLD_SIZE / 2 - 1)) / (NUM_STEPS - 1);
    // const bool close_enough = abs(target - col) < window;
    // on_target_count += alive && close_enough;
    // off_target_count += alive && !close_enough;
}

template<>
__device__ void FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>::finalize(
        const uint32_t& on_target_count, const uint32_t& off_target_count,
        Fitness* result) {
    *result = (100 * on_target_count) / (1 + off_target_count);
}


// ---------------------------------------------------------------------------
// StillLife
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::STILL_LIFE>::update(
        const int& step, const int& row, const int& col,
        const Cell& cell, uint32_t& static_cells, uint32_t& live_cells) {
    if (step < NUM_STEPS - 2) return;

    const bool alive = (cell == Cell::ALIVE);

    if (step == NUM_STEPS - 2) {
        static_cells = alive;
    } else {  // step == NUM_STEPS - 1
        static_cells = static_cells && alive;
        live_cells = alive;
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::STILL_LIFE>::finalize(
        const uint32_t& static_cells, const uint32_t& live_cells,
        Fitness* result) {
    *result = (100 * static_cells) / (1 + live_cells - static_cells);
}


// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::SYMMETRY>::update(
        const int& step, const int& row, const int& col,
        const Frame& frame, uint32_t& symmetries, uint32_t& assymmetries) {
    if (step < NUM_STEPS - 1) return;

    const bool alive = frame[row][col] == Cell::ALIVE;
    const bool v_mirror = frame[row][WORLD_SIZE - 1 - col] == Cell::ALIVE;
    const bool h_mirror = frame[WORLD_SIZE - 1 - row][col] == Cell::ALIVE;
    symmetries = int(alive && h_mirror) + int(alive && v_mirror);
    assymmetries = int(alive && !h_mirror) + int(alive && !v_mirror);
}

template<>
__device__ void FitnessObserver<FitnessGoal::SYMMETRY>::finalize(
        const uint32_t& symmetries, const uint32_t& assymmetries,
        Fitness* result) {
    *result = (100 * symmetries) / (1 + assymmetries);
}


// ---------------------------------------------------------------------------
// ThreeCycle
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::THREE_CYCLE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    if (step < NUM_STEPS - 6) return;

    history = history << 1 | (cell == Cell::ALIVE);

    // On the last iteration, do some post-processing on the data to simplify
    // the reduction step later.
    if (step == NUM_STEPS - 1) {
        constexpr int mask = 0b111;
        const int pattern = history & mask;
        cycling =
            pattern != 0b000 &&
            pattern != 0b111 &&
            (history >> 3 & mask) == pattern;
        // TODO: Punishing the organism for live cells at the end helps, but can
        // provide a weak signal. Only punishing live cells that aren't repeating
        // can be better, but encourages lame solutions. Perhaps there's a better
        // way to do this?
        // Forget all the history except for the cell value on the last step,
        // so we can count up how many cells in total are alive.
        history &= 0b1;
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::THREE_CYCLE>::finalize(
        const uint32_t& live_cells, const uint32_t& cycling, Fitness* result) {
    *result = (100 * cycling) / (1 + live_cells);
}


// ---------------------------------------------------------------------------
// TwoCycle
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    if (step < NUM_STEPS - 4) return;

    history = history << 1 | (cell == Cell::ALIVE);

    // On the last iteration, do some post-processing on the data to simplify
    // the reduction step later.
    if (step == NUM_STEPS - 1) {
        constexpr int mask = 0b11;
        const int pattern = history & mask;
        cycling =
            pattern != 0b00 &&
            pattern != 0b11 &&
            (history >> 2 & mask) == pattern;
        // TODO: Punishing the organism for live cells at the end helps, but can
        // provide a weak signal. Only punishing live cells that aren't repeating
        // can be better, but encourages lame solutions. Perhaps there's a better
        // way to do this?
        // Forget all the history except for the cell value on the last step,
        // so we can count up how many cells in total are alive.
        history &= 0b1;
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::finalize(
        const uint32_t& live_cells, const uint32_t& cycling, Fitness* result) {
    *result = (100 * cycling) / (1 + live_cells);
}


// Make sure we actually instantiate a version of the class for every goal.
template class FitnessObserver<FitnessGoal::EXPLODE>;
template class FitnessObserver<FitnessGoal::GLIDERS>;
template class FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>;
template class FitnessObserver<FitnessGoal::STILL_LIFE>;
template class FitnessObserver<FitnessGoal::SYMMETRY>;
template class FitnessObserver<FitnessGoal::THREE_CYCLE>;
template class FitnessObserver<FitnessGoal::TWO_CYCLE>;

} // namespace epigenetic_gol_kernel
