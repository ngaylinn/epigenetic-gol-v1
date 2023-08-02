#include "fitness.cuh"

#include <cstdint>

#include <cub/cub.cuh>

#include "environment.h"

namespace epigenetic_gol_kernel {

/*
TODO: Ideas for additional fitness goals:

* Active: Most cells changing value in the last N frames. This feels like an
  odd thing to optimize for, but from previous experiments it produces striking
  results.

* Rebound: Fewest cells alive at time N/2, most alive at time N. This may be a
  difficult goal, but would be dramatic and quite different from the others.

* Circle: draw a circle / ring. Not really GOL related, but shows the power of
  the PhenotypeProgram to explore arbitrary bitmaps.

* ASC: "Algorithmic Specified Complexity" would interesting. It's a measure of
  how structured a pattern is, and how unlikely it is to occur naturally. More
  or less it's just the ratio of the complexity of an algorithm and its output.
  We can directly measure the complexity of a PhenotypeProgram, and could
  estimate the information content of a GOL program by trying to compress the
  last frame of the video, perhaps using the nvCOMP library.
*/

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
                // Observe this cell and store incremental fitness data in
                // scratch_a[i] and scratch_b[i]. The meaning of these two values
                // is goal-specific.
                update(step, row, col+i, local[i], scratch_a[i], scratch_b[i]);
                break;

            case FitnessGoal::GLIDERS:
            case FitnessGoal::SYMMETRY:
                // Observe this cell and store incremental fitness data in
                // scratch_a[i] and scratch_b[i]. The meaning of these two values
                // is goal-specific.
                update(step, row, col+i, global, scratch_a[i], scratch_b[i]);
                break;

            default:
                break;
        }
    }
}

template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::reduce(Fitness* result) {
    // Add up the values of scratch_a and scratch_b across all threads
    // (ie, the full GOL board). 
    auto reduce = cub::BlockReduce<uint32_t, THREADS_PER_BLOCK>();
    uint32_t sum_a = reduce.Sum(scratch_a);
    // Needed because both calls to Sum share the same workspace memory.
    __syncthreads();
    uint32_t sum_b = reduce.Sum(scratch_b);

    // CUB returns the final reduction value in thread 0. Use a goal-specific
    // function to translate the two values in sum_a and sum_b into a single
    // fitness score.
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
        // TODO: Looking at JUST the last step encourages "vertical line"
        // patterns that are chaotic and just happen to clear the world in the
        // last frame, which isn't really what we want. Maybe just averaging
        // the last few frames would be better?
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
        const Frame& frame, uint32_t& history, uint32_t& in_spaceship) {
    // This function can be used to look for any spaceship. It's configured to
    // recognize gliders (that move one space diagonally every four steps).
    constexpr int row_delta = 1;
    constexpr int col_delta = 1;
    constexpr int time_delta = 4;
    // For looking up what value this cell had time_delta steps ago.
    constexpr int last_cycle_mask = 0b1 << time_delta;    // == 0b10000
    // For looking up what values this cell had in the past time_delta steps.
    constexpr int full_period_mask = last_cycle_mask - 1; // == 0b01111

    if (step < NUM_STEPS - time_delta - 1) return;

    const Cell& cell = frame[row][col];

    history = history << 1 | (cell == Cell::ALIVE);
    if (step < NUM_STEPS - 1) return;

    const bool last_cycle = history & last_cycle_mask;
    const bool this_cycle =
        (row + row_delta < WORLD_SIZE) &&
        (col + col_delta < WORLD_SIZE) &&
        (frame[row + row_delta][col + col_delta] == Cell::ALIVE);
    const bool always_alive = (history & full_period_mask) == full_period_mask;
    // If this cell is alive and so was the matching cell from last cycle but
    // did not just hold the same value the whole time, then this cell might be
    // part of a spaceship with the parameters set above.
    // TODO: Wait, this is unused? Something must be broken here.
    in_spaceship = last_cycle && this_cycle && !always_alive;

    // If this cell was alive at some point in the last time_delta steps but
    // did not contribute to a repeating pattern, then this is garbage not
    // participating in a spaceship.
    history = (!last_cycle || !this_cycle) && (history & full_period_mask);
}

template<>
__device__ void FitnessObserver<FitnessGoal::GLIDERS>::finalize(
        const uint32_t& not_in_spaceship, const uint32_t& in_spaceship,
        Fitness* result) {
    // TODO: This goal only ever seems to produce single gliders. What can we
    // do to increase the value of multiple gliders?
    *result = (100 * in_spaceship) / (1 + not_in_spaceship);
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
        const bool repeat = (history >> 3 & mask) == pattern;
        // If the last three steps were the same as the three steps before that
        // AND the pattern wasn't just static, then this cell is cycling.
        cycling = repeat && pattern != 0b000 && pattern != 0b111;
        // Overwrite history with a count of cells that aren't cycling. That
        // is, any cell that is not repeating and was alive in at least one of
        // the last three steps.
        history = !repeat && pattern != 0b000;
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::THREE_CYCLE>::finalize(
        const uint32_t& not_cycling, const uint32_t& cycling, Fitness* result) {
    *result = (100 * cycling) / (1 + not_cycling);
}


// ---------------------------------------------------------------------------
// TwoCycle
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    if (step < NUM_STEPS - 4) return;

    const bool alive = (cell == Cell::ALIVE);
    history = history << 1 | alive;

    // On the last iteration, do some post-processing on the data to simplify
    // the reduction step later.
    if (step == NUM_STEPS - 1) {
        constexpr int mask = 0b11;
        const int pattern = history & mask;
        const bool repeat = (history >> 2 & mask) == pattern;
        // If the last two steps were the same as the two steps before that AND
        // the pattern wasn't just static, then this cell is cycling.
        cycling = repeat && pattern != 0b00 && pattern != 0b11;
        // Overwrite history with a count of cells that aren't cycling. That
        // is, any cell that is not repeating and was alive in at least one of
        // the last two steps.
        history = !repeat && pattern != 0b00;
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::finalize(
        const uint32_t& not_cycling, const uint32_t& cycling, Fitness* result) {
    *result = (100 * cycling) / (1 + not_cycling);
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
