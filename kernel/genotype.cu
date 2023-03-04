#include "genotype.cuh"

//#include <algorithm>
//#include <limits>
//#include <numeric>
//#include <random>

namespace epigenetic_gol_kernel {
namespace {

constexpr float CROSSOVER_RATE = 0.6;
constexpr float MUTATION_RATE = 0.001;

__device__ bool coin_flip(curandState* rng, float probability=0.5) {
    return curand_uniform(rng) <= probability;
}

__global__ void RandomizeKernel(
        unsigned int population_size,
        Genotype* curr_gen_genotypes,
        curandState* rngs) {
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index > population_size) return;

    curandState rng = rngs[population_index];

    curr_gen_genotypes[population_index].init_random(&rng);

    rngs[population_index] = rng;
}

__global__ void ReproductionKernel(
        unsigned int population_size,
        const unsigned int* parent_selections,
        const unsigned int* mate_selections,
        Genotype* genotypes,
        curandState* rngs) {
    const int& population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index > population_size) return;

    const int& parent_index = parent_selections[population_index];
    const Genotype& parent_genotype = genotypes[parent_index];
    const int& mate_index = mate_selections[population_index];
    const Genotype& mate_genotype = genotypes[mate_index];
    curandState rng = rngs[population_index];
    Genotype child_genotype;

    // Consider drawing genes from mate in addition to parent.
    const bool should_crossover = coin_flip(&rng, CROSSOVER_RATE);
    const bool valid_mate = (mate_index != parent_index);
    if (should_crossover && valid_mate) {
        child_genotype.init_sexual(parent_genotype, mate_genotype, &rng);
    } else {
        // Otherwise, draw genes only from parent.
        child_genotype.init_asexual(parent_genotype, &rng);
    }
    
    // Wait for all the new genotypes to be completed before overwriting any of
    // the old genotypes.
    __syncthreads();
    genotypes[population_index] = child_genotype;
    rngs[population_index] = rng;
}

} // namespace

void randomize_population(
        unsigned int population_size, Genotype* genotypes, curandState* rngs) {
    RandomizeKernel<<<
        population_size + MAX_THREADS - 1 / MAX_THREADS,
        min(population_size, MAX_THREADS)
    >>>(population_size, genotypes, rngs);
}

void breed_population(
        unsigned int population_size,
        const unsigned int* parent_selections,
        const unsigned int* mate_selections,
        Genotype* genotypes,
        curandState* rngs) {
    ReproductionKernel<<<
        (population_size + MAX_THREADS - 1) / MAX_THREADS,
        min(population_size, MAX_THREADS)
    >>>(population_size, parent_selections, mate_selections, genotypes, rngs);
}

__device__ void Genotype::init_random(curandState* rng) {
    for (auto& scalar: scalars) {
        scalar = curand(rng);
    }
    for (auto& stamp: stamps) {
        float density = curand_uniform(rng);
        for (auto& row: stamp) {
            for (auto& cell: row) {
                cell = coin_flip(rng, density) ? Cell::ALIVE : Cell::DEAD;
            }
        }
    }
}

__device__ void Genotype::init_asexual(const Genotype& parent, curandState* rng) {
    memcpy(scalars, parent.scalars, sizeof(scalars));
    memcpy(stamps, parent.stamps, sizeof(stamps));
    mutate(rng);
}

__device__ void Genotype::init_sexual(
        const Genotype& parent, const Genotype& mate, curandState* rng) {
    Genotype const* source_genotype = nullptr;
    for (int i = 0; i < NUM_SCALARS; i++) {
        source_genotype = coin_flip(rng) ? &parent : &mate;
        scalars[i] = source_genotype->scalars[i];
    }
    for (int i = 0; i < NUM_STAMPS; i++) {
        // Randomly determine crossover behavior.
        const bool crossover_axis = coin_flip(rng);
        const bool crossover_order = coin_flip(rng);
        for (int row = 0; row < STAMP_SIZE; row++) {
            for (int col = 0; col < STAMP_SIZE; col++) {
                // Split the genotype down the middle, either horizontally or
                // vertically. Which side does this cell fall on?
                const bool side = (crossover_axis ? col : row) < STAMP_SIZE / 2;
                // Half the genotype data will come from parent, the other half
                // from mate. Which half is which?
                if (crossover_order) {
                    source_genotype = side ? &parent : &mate;
                } else {
                    source_genotype = side ? &mate : &parent;
                }
                stamps[i][row][col] =
                    source_genotype->stamps[i][row][col];
            }
        }
    }
    mutate(rng);
}

__device__ void Genotype::mutate(curandState* rng) {
    for (auto& scalar: scalars) {
        if (coin_flip(rng, MUTATION_RATE)) {
            scalar = curand(rng);
        }
    }
    for (auto& stamp: stamps) {
        for (auto& row: stamp) {
            for (auto& cell: row) {
                if (coin_flip(rng, MUTATION_RATE)) {
                    cell = coin_flip(rng) ? Cell::ALIVE : Cell::DEAD;
                }
            }
        }
    }
}

__device__ unsigned int Genotype::get_scalar(unsigned int index) const {
    return scalars[index];
}

__device__ const Stamp& Genotype::get_stamp(unsigned int index) const {
    return stamps[index];
}

} // namespace epigenetic_gol_kernel
