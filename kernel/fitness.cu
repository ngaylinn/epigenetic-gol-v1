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

TODO: What if you applied different goals to different regions of the board?
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
            case FitnessGoal::EXPLODE:
            case FitnessGoal::LEFT_TO_RIGHT:
            case FitnessGoal::RING:
            case FitnessGoal::STILL_LIFE:
            case FitnessGoal::THREE_CYCLE:
            case FitnessGoal::TWO_CYCLE:
                // Observe this cell and store incremental fitness data in
                // scratch_a[i] and scratch_b[i]. The meaning of these two values
                // is goal-specific.
                update(step, row, col+i, local[i], scratch_a[i], scratch_b[i]);
                break;

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
        uint32_t& alive_at_start, uint32_t& alive_at_end) {
    if (step > 0 && step < NUM_STEPS - 1) return;

    if (step == 0) {
        alive_at_start = (cell == Cell::ALIVE);
    } else { // step == NUM_STEPS - 1
        alive_at_end += (cell == Cell::ALIVE);
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::EXPLODE>::finalize(
        const uint32_t& alive_at_start, const uint32_t& alive_at_end,
        Fitness* result) {
    *result = (100 * alive_at_end) / (1 + alive_at_start);
}


// ---------------------------------------------------------------------------
// LeftToRight
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& on_target_first_frame, uint32_t& on_target_last_frame) {
    if (step > 0 && step < NUM_STEPS - 1) return;

    const bool alive = (cell == Cell::ALIVE);

    // On the first step, a cell is "on target" if it is ALIVE on the left or
    // DEAD on the right. The opposite is true for the last step.
    if (step == 0) {
        on_target_first_frame = alive == (col < WORLD_SIZE / 2);
    } else if (step == NUM_STEPS - 1) {
        on_target_last_frame = alive == (col >= WORLD_SIZE / 2);
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>::finalize(
        const uint32_t& on_target_first_frame,
        const uint32_t& on_target_last_frame,
        Fitness* result) {
    // Look for simulations with high on-target values for the first and last
    // steps. Weight the last step higher than the first, since it's much
    // easier to craft a good starting phenotype than to have the simulation
    // ultimately produce a good last step.
    *result = on_target_first_frame + 2 * on_target_last_frame;
    //*result = (100 * on_target_count) / (1 + off_target_count);
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

namespace {
template<int CYCLE_LENGTH>
__device__ void update_cycle(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    // How many times must the cycle repeat in order to count it?
    constexpr int NUM_ITERATIONS = 4;
    // A bitmask to capture the last CYCLE_LENGTH bits.
    constexpr int MASK = (2 << (CYCLE_LENGTH - 1)) - 1;

    // Only consider the last few steps of the simulation, just enough to
    // capture the desired number of cycles.
    if (step < NUM_STEPS - CYCLE_LENGTH * NUM_ITERATIONS) return;

    // Record a history of this cell's state, one bit per simulation step.
    history = history << 1 | (cell == Cell::ALIVE);

    // On the last iteration, test if this cell was oscillating the whole time.
    if (step < NUM_STEPS - 1) return;

    // Capture the pattern found in the last N steps of the simulation and
    // figure out if this cell stayed on, off, or changed in that time.
    const int last_cycle = history & MASK;
    cycling = 1;
    bool not_cycling = false;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // Grab the last N states from this cycle from the history variable,
        // then bitshift to set up the next N for the next loop iteration.
        const int this_cycle = history & MASK;
        history >>= CYCLE_LENGTH;

        const bool always_off = (this_cycle == 0);
        const bool always_on = (this_cycle == MASK);
        const bool repeating = (this_cycle == last_cycle);
        // True iff this cell is in a repeating, non-static pattern.
        cycling &= !always_off && !always_on && repeating;
        // True for any cell that was on at some point but was not repeating.
        not_cycling |= !(always_off || repeating);
    }
    // Reuse the history variable to track not_cycling.
    history = not_cycling;
}
} // namespace

template<>
__device__ void FitnessObserver<FitnessGoal::THREE_CYCLE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    update_cycle<3>(step, row, col, cell, history, cycling);
}

template<>
__device__ void FitnessObserver<FitnessGoal::THREE_CYCLE>::finalize(
        const uint32_t& not_cycling, const uint32_t& cycling, Fitness* result) {
    // Prefer simulations with more cells cycling, and with relatively little
    // "debris" that's not participating.
    *result = (cycling * cycling) / (1 + not_cycling);
}


// ---------------------------------------------------------------------------
// TwoCycle
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    update_cycle<2>(step, row, col, cell, history, cycling);
}

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::finalize(
        const uint32_t& not_cycling, const uint32_t& cycling, Fitness* result) {
    // Prefer simulations with more cells cycling, and with relatively little
    // "debris" that's not participating.
    *result = (cycling * cycling) / (1 + not_cycling);
}


// ---------------------------------------------------------------------------
// Ring
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::RING>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& on_target, uint32_t& off_target) {
    if (step < NUM_STEPS - 1) return;

    constexpr int CENTER = WORLD_SIZE / 2;
    constexpr int INNER_RADIUS = CENTER / 4;
    constexpr int OUTER_RADIUS = 3 * CENTER / 4;

    int distance_squared = (
        (CENTER - row) * (CENTER - row) +
        (CENTER - col) * (CENTER - col));
    bool within_target = (
        (distance_squared > INNER_RADIUS * INNER_RADIUS) &&
        (distance_squared <= OUTER_RADIUS * OUTER_RADIUS));

    on_target = within_target == (cell == Cell::ALIVE);
    off_target = !on_target;
}

template<>
__device__ void FitnessObserver<FitnessGoal::RING>::finalize(
        const uint32_t& on_target, const uint32_t& off_target, Fitness* result) {
    *result = (100 * on_target) / (1 + off_target);
}


// ---------------------------------------------------------------------------

// Make sure we actually instantiate a version of the class for every goal.
template class FitnessObserver<FitnessGoal::EXPLODE>;
template class FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>;
template class FitnessObserver<FitnessGoal::RING>;
template class FitnessObserver<FitnessGoal::STILL_LIFE>;
template class FitnessObserver<FitnessGoal::SYMMETRY>;
template class FitnessObserver<FitnessGoal::THREE_CYCLE>;
template class FitnessObserver<FitnessGoal::TWO_CYCLE>;

} // namespace epigenetic_gol_kernel
