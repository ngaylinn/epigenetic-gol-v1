#ifndef __SELECTION_H__
#define __SELECTION_H__

#include <vector>
#include <curand_kernel.h>

#include "environment.h"

namespace epigenetic_gol_kernel {

/*
 * Perform selection on the whole population.
 *
 * Note, this function isn't very efficient. It runs on the GPU since that's
 * where the data is, but only uses a small fraction of its power.
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
 * implementation, though they use different sources of randomness and may
 * produce inconsistent results.
 */
std::vector<unsigned int> select(
		const std::vector<Fitness> scores,
		const unsigned int random_value);

} // namespace epigenetic_gol_kernel

#endif
