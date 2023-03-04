#ifndef __GOL_H__
#define __GOL_H__

#include "environment.h"
#include "genotype.cuh"
#include "interpreter.cuh"

namespace epigenetic_gol_kernel {

/*
 * Compute Game of Life Simulations in parallel.
 * 
 * This function simulates population_size individual Game of Life simulations,
 * running each for NUM_STEPS iterations. Fitness is computed based on the
 * given goal. Videos are recorded only on request.
 *
 * Do not call this function directly, use the Simulator class instead.
 */
void simulate_population(
        const unsigned int population_size,
        const unsigned int num_species,
        const Interpreter* const* interpreters,
        const Genotype* genotypes,
        const FitnessGoal& goal,
        Video* videos,
        Fitness* fitness_scores,
        bool record = false);

} // namespace epigenetic_gol_kernel

#endif
