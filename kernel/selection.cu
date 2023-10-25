#include "selection.h"

#include <thrust/swap.h>
#include "cuda_utils.cuh"

namespace epigenetic_gol_kernel {
namespace {

/*
 * Choose batch_size individuals randomly, in proportion to fitness.
 *
 * This function uses Stochastic Universal Sampling. This randomly selects
 * individuals proportionate to fitness, but avoids statistically unlikely
 * scenarios (like choosing the same individual count times) by basing all
 * selections off of a single random value.
 *
 * Note that this function is written to work on either the GPU or the CPU.
 * This makes it possible for the Python code to reuse the same implementation
 * to select species as the C++ code uses to select organisms.
 */
__device__ __host__ void select(
        const unsigned int batch_size,
        const unsigned int batch_offset,
        const Fitness* fitness_scores,
        const unsigned int random_value,
        unsigned int* selections) {
    // Imagine a roulette wheel, where each individual is assigned a wedge that
    // is proportional to its fitness. The full circumference of that wheel is
    // the total fitness for the population. Note that while Fitness values are
    // integers, this computation uses floats to avoid edge cases when values
    // don't divide evenly.
    float total_fitness = 0;
    for (int i = 0; i < batch_size; i++) {
        total_fitness += fitness_scores[i];
    }
    // If all organisms scored a 0, the algorithm below wouldn't work. Since
    // they're all equally (un)fit, just pick each individual one time.
    if (total_fitness == 0) {
        for (int i = 0; i < batch_size; i++) {
            // Note that we add batch_offset here to switch from batch-relative
            // indexing to population-relative indexing.
            selections[i] = batch_offset + i;
        }
        return;
    }

    // Pick batch_size equidistant sampling points around the edge of that
    // roulette wheel, starting at a random location.
    const float sample_period = total_fitness / batch_size;
    // The random_value passed in needs special handling to avoid overflow.
    const float sample_offset = float(
        double(random_value) * sample_period / double(long(1)<<32));

    // Walk around the edge of the roulette wheel to figure out which wedge
    // contains each sample point. The individual corresponding to that wedge /
    // sample point will be selected. More fit individuals have bigger wedges
    // which may contain multiple sample points, which means that individual
    // may get selected more than once and can have multiple offspring.
    // Individuals with a fitness score smaller than the sample_period may fall
    // between sample points, in which case they won't be selected and won't
    // pass on their genes. The selection_index variable is an index into the
    // batch of organisms to select from / the wedges of the roulette wheel.
    // Starting at -1 indicates we have not yet reached a sample point inside
    // the 0th wedge, but will do so in the first iteration of the loop below.
    int selection_index = -1;
    Fitness fitness_so_far = 0;
    for (int i = 0; i < batch_size; i++) {
        float sample_point = sample_offset + i * sample_period;
        // Step through the wedges one at a time to find the one that overlaps
        // with this sample point. This could happen 0 times if the last wedge
        // is so big it contains this next sample point, too, or it could
        // happen many times, if there are many thin wedges to pass over.
        while (sample_point > fitness_so_far) {
            selection_index += 1;
            fitness_so_far += fitness_scores[selection_index];
        }
        // Actually select the individual corresponding to this sample point.
        // Note that we add batch_offset here to switch from batch-relative
        // indiexing to population-relative indexing.
        selections[i] = batch_offset + selection_index;
    }
}

__global__ void SelectKernel(
        unsigned int population_size,
        unsigned int num_organisms,
        const Fitness* fitness_scores,
        unsigned int* parent_selections,
        unsigned int* mate_selections,
        curandState* rngs) {
    // The population_size argument is the total number of organisms of all
    // species, but only organisms of the same species should be paired for
    // breeding. So, divide the population into batches of num_organisms each.
    // Each thread will handle one batch of organisms of the same species.
    const int batch_offset =
        (blockIdx.x * blockDim.x + threadIdx.x) * num_organisms;
    if (batch_offset + num_organisms > population_size) return;

    // Copy RNG state to registers for faster repeated access.
    curandState rng = rngs[batch_offset];

    // Select num_organisms parents and mates, from the batch of individuals
    // starting at batch_offset.
    select(num_organisms, batch_offset, &fitness_scores[batch_offset],
            curand(&rngs[batch_offset]), &parent_selections[batch_offset]);
    select(num_organisms, batch_offset, &fitness_scores[batch_offset],
            curand(&rngs[batch_offset]), &mate_selections[batch_offset]);

    // Shuffle mates to avoid preferentially pairing organisms with others that
    // have similar fitness.
    for (int i = num_organisms - 1; i > 0; i--) {
        int j = curand(&rng) % i;
        thrust::swap(
            mate_selections[batch_offset + i],
            mate_selections[batch_offset + j]);
    }

    // Save the modified RNG state back to global memory.
    rngs[batch_offset] = rng;
}

} // namespace

void select_from_population(
        unsigned int population_size, unsigned int num_organisms,
        const Fitness* fitness_scores, unsigned int* parent_selections,
        unsigned int* mate_selections, curandState* rngs) {
    // Break the work down into batches of num_organisms individuals.
    int batches = population_size / num_organisms;
    SelectKernel<<<
        (batches + MAX_THREADS - 1) / MAX_THREADS,
        min(batches, MAX_THREADS)
    >>>(population_size, num_organisms, fitness_scores,
        parent_selections, mate_selections, rngs);
    CUDA_CHECK_ERROR();
}

std::vector<unsigned int> select(
        const std::vector<Fitness> scores,
        const unsigned int random_value) {
    int batch_size = scores.size();
    std::vector<unsigned int> result(batch_size);
    select(batch_size, 0, scores.data(), random_value, result.data());
    return result;
}

} // namespace epigenetic_gol_kernel
