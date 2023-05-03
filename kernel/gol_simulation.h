#ifndef __GOL_SIMULATION_H__
#define __GOL_SIMULATION_H__

#include "environment.h"
#include "phenotype_program.h"

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
        const FitnessGoal& goal,
        const PhenotypeProgram* programs,
        const Genotype* genotypes,
        Video* videos,
        Fitness* fitness_scores,
        bool record = false);

// Run a single Game of Life Simulation on the CPU (for testing)
Video* simulate_phenotype(const Frame& phenotype);

// Render a single phenotype (for testing)
const Frame* render_phenotype(
        const PhenotypeProgram& h_program,
        const Genotype* h_genotype=nullptr);

} // namespace epigenetic_gol_kernel

#endif
