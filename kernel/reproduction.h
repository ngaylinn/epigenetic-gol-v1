#ifndef __REPRODUCTION_H__
#define __REPRODUCTION_H__

#include <vector>

#include <curand_kernel.h>

#include "environment.h"
#include "phenotype_program.h"

namespace epigenetic_gol_kernel {

// Constants configuring reproduction behavior.
constexpr float CROSSOVER_RATE = 0.6;
constexpr float MUTATION_RATE = 0.001;

/* 
 * Initialize genotypes with random values.
 *
 * Note, this function isn't very efficient. It runs on the GPU since that's
 * where the data is, but only uses a small fraction of its power.
 *
 * Do not call this function directly, use the Simulator class instead.
 */
void randomize_population(
        unsigned int population_size,
        Genotype* genotypes,
        curandState* rngs);

/* 
 * Initialize output_genotypes from input_genotypes.
 *
 * The given selections are used for breeding, though whether an organism
 * reproduces sexually or asexually is determined by chance (the mate is
 * just an optional gene donor). Note, this function isn't very efficient. It
 * runs on the GPU since that's where the data is, but only uses a small
 * fraction of its power.
 *
 * Do not call this function directly, use the Simulator class instead.
 */
void breed_population(
        unsigned int population_size,
        const unsigned int* parent_selections,
        const unsigned int* mate_selections,
        const Genotype* input_genotypes,
        Genotype* output_genotypes,
        curandState* rngs);

// A standalone version of the above to be called directly for testing.
const Genotype* breed_population(
        const Genotype* h_input_genotypes,
        std::vector<unsigned int> h_parent_selections,
        std::vector<unsigned int> h_mate_selections);


} // namespace epigenetic_gol_kernel

#endif
