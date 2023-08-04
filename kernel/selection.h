/*
 * Functions for randomly choosing individuals from a population by fitness.
 *
 * This module provides a batched version of this operation, which
 * simultaneously selects breeding partners across many populations, and a
 * one-off version that operates on a single population.
 */

#ifndef __SELECTION_H__
#define __SELECTION_H__

#include <vector>
#include <curand_kernel.h>

#include "environment.h"

namespace epigenetic_gol_kernel {

/*
 * Perform selection for organism populations of all species at once.
 *
 * This function assumes that population_size is a multiple of num_organisms,
 * and process selections in batches of num_organisms. It uses Stochastic
 * Universal Sampling to avoid unhelpful edge cases, like selecting the same
 * organism every time.
 * Note, this function isn't very efficient. It runs on the GPU since that's
 * where the data is, but does not utilize the device's full power.
 *
 * Do not call this function directly, use the Simulator class instead.
 */
void select_from_population(
        unsigned int population_size, unsigned int num_organisms,
        const Fitness* fitness_scores, unsigned int* parent_selections,
        unsigned int* mate_selections, curandState* rng);

/*
 * A stand-alone version of select, independent of the Simulation class.
 *
 * This function runs on the CPU, but uses the same code as the GPU-based
 * implementation. Since this is typically called from Python, let the caller
 * manage their own RNG by passing in random_value manually. This is assumed
 * to be 32-bit unsigned integer value chosen from a uniform distribution.
 */
std::vector<unsigned int> select(
        const std::vector<Fitness> scores,
        const unsigned int random_value);

} // namespace epigenetic_gol_kernel

#endif
