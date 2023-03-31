#include "reproduction.h"

//#include <algorithm>
//#include <limits>
//#include <numeric>
//#include <random>

#include "cuda_utils.cuh"

namespace epigenetic_gol_kernel {
namespace {

__device__ bool coin_flip(curandState* rng, float probability=0.5) {
    return curand_uniform(rng) <= probability;
}

__global__ void RandomizeKernel(
        unsigned int population_size,
        Genotype* genotypes,
        curandState* rngs) {
    // Which organism are we working on?
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index >= population_size) return;

    // Set all four scalar genes for this organism to random values.
    curandState rng = rngs[population_index];
    Genotype& genotype = genotypes[population_index];

    for (int i = 0; i < NUM_SCALARS; i++) {
        genotype.scalar_genes[i] = curand(&rng);
    }

    for (int i = 0; i < NUM_STAMPS; i++) {
        const float density = curand_uniform(&rng);
        for (int row = 0; row < STAMP_SIZE; row++) {
            for (int col = 0; col < STAMP_SIZE; col++) {
                genotype.stamp_genes[i][row][col] =
                    coin_flip(&rng, density) ? Cell::ALIVE : Cell::DEAD;
            }
        }
    }

    rngs[population_index] = rng;
}

__global__ void ReproduceKernel(
        unsigned int population_size,
        const unsigned int* parent_selections,
        const unsigned int* mate_selections,
        const Genotype* input_genotypes,
        Genotype* output_genotypes,
        curandState* rngs) {
    // Which organism are we working on?
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index >= population_size) return;

    const int& parent_index = parent_selections[population_index];
    const int& mate_index = mate_selections[population_index];
    curandState rng = rngs[population_index];
    Genotype& output_genotype = output_genotypes[population_index];

    // Consider drawing gene values from mate instead of parent.
    const bool should_crossover = 
        mate_index != parent_index && coin_flip(&rng, CROSSOVER_RATE);

    for (int i = 0; i < NUM_SCALARS; i++) {
        // Which organism should we draw gene data from? If we're doing asexual
        // reproduction, just take from the parent. If we're doing sexual
        // reproduction, then for each scalar it has a 50% chance of coming
        // from either parent or mate.
        const int source_index =
            (should_crossover && coin_flip(&rng)) ? mate_index : parent_index;
        const Genotype& input_genotype = input_genotypes[source_index];
        // Set the gene value, either from source or a random mutation.
        output_genotype.scalar_genes[i] =
            coin_flip(&rng, MUTATION_RATE)
            ? curand(&rng) : input_genotype.scalar_genes[i];
    }

    for (int i = 0; i < NUM_STAMPS; i++) {
        // For stamps, crossover involves mixing together half of one stamp
        // gene with half of another. Here we randomly determine which halves
        // to recombine.
        const bool crossover_axis = should_crossover ? coin_flip(&rng) : false;
        const bool mate_side = should_crossover ? coin_flip(&rng) : false;
        for (int row = 0; row < STAMP_SIZE; row++) {
            for (int col = 0; col < STAMP_SIZE; col++) {
                // Split the stamp data down the middle, either horizontally or
                // vertically. Which side does this Cell fall on?
                const bool side =
                    (crossover_axis ? col : row) < (STAMP_SIZE / 2);
                // If we're doing crossover, and the Cell we're computing falls
                // on the mate's half of the Stamp, then draw data from mate.
                // Otherwise, draw data from parent.
                const int& source_index =
                    should_crossover && side == mate_side
                    ? mate_index : parent_index;
                const Genotype& input_genotype = input_genotypes[source_index];
                // Set the gene value, either from source or a random mutation.
                output_genotype.stamp_genes[i][row][col] =
                    coin_flip(&rng, MUTATION_RATE)
                    ? (Cell) (curand(&rng) & 0xFF)
                    : input_genotype.stamp_genes[i][row][col];
            }
        }

    }

    rngs[population_index] = rng;
}

} // namespace

void randomize_population(
        unsigned int population_size,
        unsigned int num_organisms,
        Genotype* genotypes,
        curandState* rngs) {
    // Arrange the grid so that there is one thread per organism which will
    // compute ALL of that organism's genes. That's probably not the most
    // efficient way to do it, but this is operation isn't a performance
    // bottleneck, and the code is much simpler when we can share the same
    // curandState object for the whole computation.
    unsigned int organisms_per_block = min(MAX_THREADS, population_size);
    RandomizeKernel<<<
        (population_size + organisms_per_block - 1) / organisms_per_block,
        organisms_per_block
    >>>(population_size, genotypes, rngs);
    CUDA_CHECK_ERROR();
}

void breed_population(
        unsigned int population_size,
        unsigned int num_organisms,
        const unsigned int* parent_selections,
        const unsigned int* mate_selections,
        const Genotype* input_genotypes,
        Genotype* output_genotypes,
        curandState* rngs) {
    // Arrange the grid so that there is one thread per organism which will
    // compute ALL of that organism's genes. That's probably not the most
    // efficient way to do it, but this is operation isn't a performance
    // bottleneck, and the code is much simpler when we can share the same
    // curandState object for the whole computation.
    unsigned int organisms_per_block = min(MAX_THREADS, population_size);
    ReproduceKernel<<<
        (population_size + organisms_per_block - 1) / organisms_per_block,
        organisms_per_block
    >>>(population_size, parent_selections, mate_selections,
            input_genotypes, output_genotypes, rngs);
    CUDA_CHECK_ERROR();
}

} // namespace epigenetic_gol_kernel
