#ifndef __GENOTYPE_H__
#define __GENOTYPE_H__

#include <curand_kernel.h>

#include "environment.h"

namespace epigenetic_gol_kernel {

constexpr unsigned int NUM_SCALARS = 4;
constexpr unsigned int NUM_STAMPS = 4;
constexpr unsigned int STAMP_SIZE = 8;
typedef Cell Stamp[STAMP_SIZE][STAMP_SIZE];

/*
 * Represents the gene sequence for one individual.
 *
 * Each gene sequence is just a string of bits. This class holds that raw data,
 * and overlays it with structure representing some number of genes of specific
 * types. This makes it easy for the Interpreter and Operations classes to
 * define and lookup values and interpret their meaning for phenotype
 * generation. It also makes it easy to generate and cross breed Genotypes
 * while respecting gene types and boundaries, even without knowing how those
 * genes will be used.
 */
// TODO: Consider eliminating this class entirely and instead just managing two
// arrays of data.
class Genotype {
    private:
        unsigned int scalars[NUM_SCALARS];
        Stamp stamps[NUM_STAMPS];

        // Add some random variation to this Genotype.
        __device__ void mutate(curandState* rng);

    public:
        // Randomly initialize a Genotype.
        __device__ void init_random(curandState* rng);

        // Generate a new Genotype derived from parent.
        __device__ void init_asexual(const Genotype& parent, curandState* rng);

        // Generate a new Genotype derived from parent and mate.
        __device__ void init_sexual(
                const Genotype& parent, const Genotype& mate, curandState* rng);

        __device__ unsigned int get_scalar(unsigned int index) const;
        __device__ const Stamp& get_stamp(unsigned int index) const;
};

/* 
 * Initialize genotypes with random values.
 *
 * Note, this function isn't very efficient. It runs on the GPU since that's
 * where the data is, but only uses a small fraction of its power.
 *
 * Do not call this function directly, use the Simulator class instead.
 */
void randomize_population(
        unsigned int population_size, Genotype* genotypes, curandState* rngs);

/* 
 * Initialize next_gen_genotypes from curr_gen_genotypes.
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
        Genotype* genotypes,
        curandState* rngs);

} // namespace epigenetic_gol_kernel

#endif
