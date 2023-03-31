#include "fitness.cuh"

namespace epigenetic_gol_kernel {
namespace {

// TODO: Add more fitness goals.
// - Center-of-mass analysis version of left-to-right, other variants.
// - Glider detection
// - Algorithmic Specified Complexity?

__device__ void update_still_life(
        const int& step, const int&, const int&,
        const Cell& cell, PartialFitness& partial_fitness) {
    Cell& prev_value = partial_fitness.cell[0];
    Cell& last_value = partial_fitness.cell[1];

    if (step < NUM_STEPS - 2) return;
    if (step == NUM_STEPS - 2) {
        prev_value = cell;
        return;
    }
    // At this point, step == NUM_STEPS - 1
    last_value = cell;
}

__device__ void finalize_still_life(PartialFitness& partial_fitness) {
    const Cell& prev_value = partial_fitness.cell[0];
    const Cell& last_value = partial_fitness.cell[1];

    partial_fitness.fitness = (
            last_value == prev_value &&
            last_value == Cell::ALIVE);
}

__device__ void update_two_cycle(
        const int& step, const int&, const int&,
        const Cell& cell, PartialFitness& partial_fitness) {
    Cell& prev_even = partial_fitness.cell[0];
    Cell& prev_odd = partial_fitness.cell[1];
    Cell& last_even = partial_fitness.cell[2];
    Cell& last_odd = partial_fitness.cell[3];

    if (step < NUM_STEPS - 4) return;
    if (step == NUM_STEPS - 4) {
        prev_even = cell;
        return;
    }
    if (step == NUM_STEPS - 3) {
        prev_odd = cell;
        return;
    }
    if (step == NUM_STEPS - 2) {
        last_even = cell;
        return;
    }
    // At this point, step == NUM_STEPS - 1
    last_odd = cell;
}

__device__ void finalize_two_cycle(PartialFitness& partial_fitness) {
    const Cell& prev_even = partial_fitness.cell[0];
    const Cell& prev_odd = partial_fitness.cell[1];
    const Cell& last_even = partial_fitness.cell[2];
    const Cell& last_odd = partial_fitness.cell[3];

    partial_fitness.fitness = (
            last_odd == prev_odd &&
            last_even == prev_even &&
            last_odd != last_even);
}

} // namespace


__device__ void update_fitness(
        const int& step, const int& row, const int& col, const Cell& cell,
        FitnessGoal goal, PartialFitness& partial_fitness) {
    switch (goal) {
        case FitnessGoal::STILL_LIFE:
            return update_still_life(step, row, col, cell, partial_fitness);
        case FitnessGoal::TWO_CYCLE:
            return update_two_cycle(step, row, col, cell, partial_fitness);
        default:
            return;
    }
}

__device__ void finalize_fitness(
        FitnessGoal goal, PartialFitness& partial_fitness) {
    switch (goal) {
        case FitnessGoal::STILL_LIFE:
            return finalize_still_life(partial_fitness);
        case FitnessGoal::TWO_CYCLE:
            return finalize_two_cycle(partial_fitness);
        default:
            return;
    }
}

} // namespace epigenetic_gol_kernel
