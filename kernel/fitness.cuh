#ifndef __FITNESS_H__
#define __FITNESS_H__

#include <array>
#include "environment.h"

namespace epigenetic_gol_kernel {

/*
 * A workspace for computing fitness incrementally.
 *
 * In this project, fitness is evaluated for each frame as it gets computed.
 * This type is used to track state. Different fitness goals have different
 * state needs, and this union is meant to represent all of them.
 *
 * As far as the caller is concerned, the data here is undefined until they
 * call finalize_fitness. Then the fitness member is guaranteed to be set, and
 * will represent a single Cell's contribution to overall fitness.
 */
union PartialFitness {
    Fitness fitness;
    Cell cell[4];
};

/*
 * Consider a single cell from a single frame and what contribution it should
 * have to overall fitness. Store working data in partial_fitness.
 */
__device__ void update_fitness(
        const int& step, const int& row, const int& col, const Cell& cell,
        FitnessGoal goal, PartialFitness& partial_fitness);

/*
 * Combine data from all observations of a Cell across every frame of the
 * simulation to a single final fitness contribution for that Cell. After
 * calling this function, the overall fitness of the simulation is the sum of
 * PartialFitness.fitness for all cells in the world grid.
 */
__device__ void finalize_fitness(
        FitnessGoal goal, PartialFitness& partial_fitness);

} // namespace epigenetic_gol_kernel

#endif
