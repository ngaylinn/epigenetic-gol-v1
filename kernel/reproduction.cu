#include "reproduction.h"

#include "cuda_utils.cuh"

namespace epigenetic_gol_kernel {
namespace {

// Generate a pseudorandom boolean value
__device__ bool coin_flip(curandState* rng, float probability=0.5) {
    return curand_uniform(rng) <= probability;
}

// Randomly initialize population_size Genotypes
__global__ void RandomizeKernel(
        unsigned int population_size,
        Genotype* genotypes,
        curandState* rngs) {
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index >= population_size) return;

    Genotype& genotype = genotypes[population_index];
    // Copy RNG state to registers for faster repeated access.
    curandState rng = rngs[population_index];

    // Set all four Scalar genes to random values.
    for (int i = 0; i < NUM_GENES; i++) {
        genotype.scalar_genes[i] = curand(&rng);
    }

    // Randomize every Cell value in every Stamp. For the sake of diversity,
    // though, generate a mix of sparse Stamps (few live Cells) and dense
    // Stamps (many live Cells).
    for (int i = 0; i < NUM_GENES; i++) {
        const float density = curand_uniform(&rng);
        for (int row = 0; row < STAMP_SIZE; row++) {
            for (int col = 0; col < STAMP_SIZE; col++) {
                genotype.stamp_genes[i][row][col] =
                    coin_flip(&rng, density) ? Cell::ALIVE : Cell::DEAD;
            }
        }
    }

    // Save the modified RNG state back to global memory.
    rngs[population_index] = rng;
}

// Generate population_size output_genotypes by recombining data from the
// input_genotypes, optionally cross breeding using the given selections.
__global__ void ReproduceKernel(
        unsigned int population_size,
        const unsigned int* parent_selections,
        const unsigned int* mate_selections,
        const Genotype* input_genotypes,
        Genotype* output_genotypes,
        curandState* rngs) {
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index >= population_size) return;

    const int& parent_index = parent_selections[population_index];
    const int& mate_index = mate_selections[population_index];
    Genotype& output_genotype = output_genotypes[population_index];
    // Copy RNG state to registers for faster repeated access.
    curandState rng = rngs[population_index];

    // Consider drawing gene values from mate instead of parent. Note that
    // selection may match an organism with itself, so handle that edge case
    // here.
    const bool should_crossover =
        mate_index != parent_index && coin_flip(&rng, CROSSOVER_RATE);

    for (int i = 0; i < NUM_GENES; i++) {
        // Which organism should we draw gene data from? If we're doing asexual
        // reproduction, just take from the parent. If we're doing sexual
        // reproduction, then for each Scalar it has a 50% chance of coming
        // from either parent or mate.
        const int source_index =
            (should_crossover && coin_flip(&rng)) ? mate_index : parent_index;
        const Genotype& input_genotype = input_genotypes[source_index];
        // Set the gene value, either from source or a random mutation.
        output_genotype.scalar_genes[i] =
            coin_flip(&rng, MUTATION_RATE)
            ? curand(&rng) : input_genotype.scalar_genes[i];
    }

    for (int i = 0; i < NUM_GENES; i++) {
        // For Stamps, crossover involves mixing together half of one Stamp
        // gene with half of another. Here we randomly determine which halves
        // to recombine.
        const bool crossover_axis = should_crossover ? coin_flip(&rng) : false;
        const bool mate_side = should_crossover ? coin_flip(&rng) : false;
        for (int row = 0; row < STAMP_SIZE; row++) {
            for (int col = 0; col < STAMP_SIZE; col++) {
                // Split the Stamp data down the middle, either horizontally or
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
                    ? (coin_flip(&rng) ? Cell::ALIVE : Cell::DEAD)
                    : input_genotype.stamp_genes[i][row][col];
            }
        }

    }

    // Save the modified RNG state back to global memory.
    rngs[population_index] = rng;
}

} // namespace

void randomize_population(
        unsigned int population_size,
        Genotype* genotypes,
        curandState* rngs) {
    // Arrange the grid so that there is one thread per organism which will
    // compute ALL of that organism's genes. That's probably not the most
    // efficient way to do it, but this operation isn't a performance
    // bottleneck, and the code is much simpler when we can share the same
    // curandState object for the whole computation.
    unsigned int organisms_per_block = min(MAX_THREADS, population_size);
    unsigned int num_blocks =
        (population_size + organisms_per_block - 1) / organisms_per_block;
    RandomizeKernel<<<
        num_blocks, organisms_per_block
    >>>(population_size, genotypes, rngs);
    CUDA_CHECK_ERROR();
}

void breed_population(
        unsigned int population_size,
        const unsigned int* parent_selections,
        const unsigned int* mate_selections,
        const Genotype* input_genotypes,
        Genotype* output_genotypes,
        curandState* rngs) {
    // Arrange the grid so that there is one thread per organism which will
    // compute ALL of that organism's genes. That's probably not the most
    // efficient way to do it, but this operation isn't a performance
    // bottleneck, and the code is much simpler when we can share the same
    // curandState object for the whole computation.
    unsigned int organisms_per_block = min(MAX_THREADS, population_size);
    unsigned int num_blocks =
        (population_size + organisms_per_block - 1) / organisms_per_block;
    ReproduceKernel<<<
        num_blocks, organisms_per_block
    >>>(population_size, parent_selections, mate_selections,
        input_genotypes, output_genotypes, rngs);
    CUDA_CHECK_ERROR();
}

const Genotype* breed_population(
        const Genotype* h_input_genotypes,
        std::vector<unsigned int> h_parent_selections,
        std::vector<unsigned int> h_mate_selections) {
    // This is a one-off operation, not a batch one, but there is no CPU
    // version of the reproduction process, so it still must be run on the GPU.
    // That's pretty inefficient, but fine for testing. Since Simulator isn't
    // providing any memory allocations, do that manually.
    const int population_size = h_parent_selections.size();
    DeviceData<Genotype> input_genotypes(population_size, h_input_genotypes);
    DeviceData<Genotype> output_genotypes(population_size);
    DeviceData<unsigned int> parent_selections(
            population_size, h_parent_selections.data());
    DeviceData<unsigned int> mate_selections(
            population_size, h_mate_selections.data());
    DeviceData<curandState> rngs(population_size);

    // Seed the RNGs to the same value every time for consistent results.
    seed_rngs(rngs, population_size, 42);

    breed_population(
            population_size, parent_selections, mate_selections,
            input_genotypes, output_genotypes, rngs);

    return output_genotypes.copy_to_host();
}

} // namespace epigenetic_gol_kernel
