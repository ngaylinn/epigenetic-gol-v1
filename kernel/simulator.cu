#include "environment.h"
#include "phenotype_program.h"
#include "simulator.h"

// TODO: Unused?
#include <vector>

#include "cuda_utils.cuh"
#include "gol_simulation.h"
#include "reproduction.h"
#include "selection.h"

namespace epigenetic_gol_kernel {

// A convenience class for allocating, initializing, and freeing all of
// the GPU-side data objects used by Simulation.
class Simulator::DeviceAllocations {
    friend class Simulator;
    private:
        DeviceData<curandState> rngs;
        DeviceData<PhenotypeProgram> programs;
        DeviceData<Genotype> curr_gen_genotypes;
        DeviceData<Genotype> next_gen_genotypes;
        DeviceData<unsigned int> parent_selections;
        DeviceData<unsigned int> mate_selections;
        DeviceData<Fitness> fitness_scores;
        // The Videos take up a ton of space on the GPU, and aren't necessary
        // most of the time, since videos are typically only recorded on
        // demand. This was added to simplify the ENTROPY fitness goal, which
        // requires capturing full simulation videos.
        DeviceData<Video> videos;

        DeviceAllocations(int num_species, int size)
            : programs(num_species),
              rngs(size),
              curr_gen_genotypes(size),
              next_gen_genotypes(size),
              parent_selections(size),
              mate_selections(size),
              fitness_scores(size),
              videos(size) {
        }
};

Simulator::Simulator(
        unsigned int num_species,
        unsigned int num_trials,
        unsigned int num_organisms)
    : num_species(num_species),
      num_trials(num_trials),
      num_organisms(num_organisms),
      size(num_species * num_trials * num_organisms) {
    d = new DeviceAllocations(num_species, size);
    // Normally the caller would seed the Simulator manually, but convenience
    // and safety, make sure the RNGs always get initialized to SOMETHING.
    seed(42);
}

Simulator::~Simulator() {
    delete d;
}


// ---------------------------------------------------------------------------
// Methods for managing simulation control flow
// ---------------------------------------------------------------------------

void Simulator::populate(const PhenotypeProgram* h_programs) {
    d->programs.copy_from_host(h_programs);
    randomize_population(size, d->curr_gen_genotypes, d->rngs);
}

void Simulator::propagate() {
    select_from_population(size, num_organisms, d->fitness_scores,
            d->parent_selections, d->mate_selections, d->rngs);

    breed_population(
            size, d->parent_selections, d->mate_selections,
            d->curr_gen_genotypes, d->next_gen_genotypes, d->rngs);

    // After generating next_gen_genotypes from curr_gen_genotypes, swap the
    // two pointers so that the the new data is considered current and the old
    // data may be overwritten to compute the next generation.
    d->curr_gen_genotypes.swap(d->next_gen_genotypes);
}

void Simulator::simulate(const FitnessGoal& goal) {
    // Passing in d->videos is only necessary for the ENTROPY FitnessGoal, but
    // it's simpler just to use it every time.
    simulate_population<false>(
            size, num_species, goal, d->programs,
            d->curr_gen_genotypes, d->fitness_scores, d->videos);
}

Video* Simulator::simulate_and_record(const FitnessGoal& goal) {
    simulate_population<true>(
            size, num_species, goal, d->programs,
            d->curr_gen_genotypes, d->fitness_scores, d->videos);
    return d->videos.copy_to_host();
}

const Fitness* Simulator::evolve(
        const PhenotypeProgram* h_programs,
        const FitnessGoal& goal,
        const int num_generations) {
    populate(h_programs);
    for (int i = 0; i < num_generations - 1; i++) {
        simulate(goal);
        propagate();
    }
    simulate(goal);
    return get_fitness_scores();
}


// ---------------------------------------------------------------------------
// Methods to retrieve simulation results computed by simulate()
// ---------------------------------------------------------------------------

const Fitness* Simulator::get_fitness_scores() const {
    return d->fitness_scores.copy_to_host();
}

const Genotype* Simulator::get_genotypes() const {
    return d->curr_gen_genotypes.copy_to_host();
}


// ---------------------------------------------------------------------------
// Methods to manage RNG state
// ---------------------------------------------------------------------------

void Simulator::seed(const unsigned int seed_value) {
    seed_rngs(d->rngs, size, seed_value);
}

} // namespace epigenetic_gol_kernel


// ---------------------------------------------------------------------------
// A pure C++ demo for use with NVidia's profiling and debug tools.
// ---------------------------------------------------------------------------

#include "nvcomp.h"
#include "nvcomp/cascaded.h"

int main(int argc, char* argv[]) {
    using namespace epigenetic_gol_kernel;
    constexpr int NUM_SAMPLES = 10;
    constexpr int SAMPLE_BYTES = 4096;
    // Cascaded compression needs just 8 additional bytes overhead, so in the
    // worst case scenario compressed size is 8 bytes larger than uncompressed.
    constexpr int COMPRESSED_BYTES = 4096 + 8;

    // Step 1: Initialize host-side data.
    // Allocate host-side buffers.
    unsigned char h_uncompressed[NUM_SAMPLES][SAMPLE_BYTES] = { 0 };
    unsigned char h_compressed[NUM_SAMPLES][COMPRESSED_BYTES] = { 0 };
    unsigned char h_decompressed[NUM_SAMPLES][SAMPLE_BYTES] = { 0 };
    size_t h_uncompressed_bytes[NUM_SAMPLES] = { 0 };
    size_t h_compressed_bytes[NUM_SAMPLES] = { 0 };
    size_t h_decompressed_bytes[NUM_SAMPLES] = { 0 };
    // Generate sample data to compress.
    // The first three samples are a mix of two values in contiguous blocks.
    for (int i = 0; i < 3; i++) {
        for (int offset = 0; offset < SAMPLE_BYTES; offset++) {
            h_uncompressed[i][offset] = (offset > (i + 1) * 1365) ? 0 : 42;
        }
    }
    // The next three samples cycle between a values.
    for (int i = 3; i < 6; i++) {
        for (int offset = 0; offset < SAMPLE_BYTES; offset++) {
            h_uncompressed[i][offset] = (offset % (i - 1)) * 64;
        }
    }
    // The rest of the samples are zeroes with some values randomized.
    for (int i = 6; i < NUM_SAMPLES; i++) {
        for (int offset = 0; offset < SAMPLE_BYTES; offset++) {
            if (rand() % 4 >= NUM_SAMPLES - i - 1) {
                h_uncompressed[i][offset] = rand() & 0xFF;
            }
        }
    }
    // All of the original samples are the same size.
    for (int i = 0; i < NUM_SAMPLES; i++) {
        h_uncompressed_bytes[i] = SAMPLE_BYTES;
    }

    // Step 2: Set up data and scratch space on the GPU.
    // Allocate device-side buffers.
    DeviceData<unsigned char> d_uncompressed(
            NUM_SAMPLES * SAMPLE_BYTES, (unsigned char*) h_uncompressed);
    DeviceData<unsigned char> d_compressed(NUM_SAMPLES * COMPRESSED_BYTES);
    DeviceData<unsigned char> d_decompressed(NUM_SAMPLES * SAMPLE_BYTES);
    DeviceData<size_t> d_uncompressed_bytes(NUM_SAMPLES, h_uncompressed_bytes);
    DeviceData<size_t> d_compressed_bytes(NUM_SAMPLES);
    DeviceData<size_t> d_decompressed_bytes(NUM_SAMPLES);

    // Compute pointer arrays based on the device-side memory locations.
    void* h_uncompressed_ptrs[NUM_SAMPLES];
    void* h_compressed_ptrs[NUM_SAMPLES];
    void* h_decompressed_ptrs[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++) {
        h_uncompressed_ptrs[i] = &(d_uncompressed[i * SAMPLE_BYTES]);
        h_compressed_ptrs[i] = &(d_compressed[i * COMPRESSED_BYTES]);
        h_decompressed_ptrs[i] = &(d_decompressed[i * SAMPLE_BYTES]);
        printf("%p\t%p\t%p\n",
                h_uncompressed_ptrs[i],
                h_compressed_ptrs[i],
                h_decompressed_ptrs[i]);
    }
    DeviceData<void*> d_uncompressed_ptrs(NUM_SAMPLES, h_uncompressed_ptrs);
    DeviceData<void*> d_compressed_ptrs(NUM_SAMPLES, h_compressed_ptrs);
    DeviceData<void*> d_decompressed_ptrs(NUM_SAMPLES, h_decompressed_ptrs);

    // Allocate temp workspace needed for compression.
    const nvcompBatchedCascadedOpts_t options = {
        COMPRESSED_BYTES, NVCOMP_TYPE_CHAR, 2, 1, 1};

    // Step 3: Compress the data and copy results to host.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    nvcompStatus_t status = nvcompBatchedCascadedCompressAsync(
            d_uncompressed_ptrs,  // input
            d_uncompressed_bytes, // input
            SAMPLE_BYTES,         // not used
            NUM_SAMPLES,          // input
            NULL,                 // not used
            0,                    // not used
            d_compressed_ptrs,    // output
            d_compressed_bytes,   // output
            options,              // input
            stream);              // input
    cudaDeviceSynchronize();
    d_compressed.copy_to_host((unsigned char*) h_compressed);
    d_compressed_bytes.copy_to_host(h_compressed_bytes);

    // Step 4: Decompress the compressed data to verify correctness.
    DeviceData<nvcompStatus_t> d_statuses(NUM_SAMPLES);
    status = nvcompBatchedCascadedDecompressAsync(
            d_compressed_ptrs,
            d_compressed_bytes,
            d_uncompressed_bytes,
            d_decompressed_bytes,
            NUM_SAMPLES,
            NULL,
            0,
            d_uncompressed_ptrs,
            d_statuses,
            stream);
    cudaDeviceSynchronize();
    // TODO: For some reason, I seem to be getting back all zeroes after
    // decompression.
    d_decompressed.copy_to_host((unsigned char*) h_decompressed);
    d_decompressed_bytes.copy_to_host(h_decompressed_bytes);

    // Step 5: Analyze results
    printf("Sample\tUncSize\tCmpSize\tRatio\tDecSize\tErrors\n");
    for (int i = 0; i < NUM_SAMPLES; i++) {
        float ratio = ((float) h_compressed_bytes[i]) / ((float) SAMPLE_BYTES);
        int errors = 0;
        for (int offset = 0; offset < SAMPLE_BYTES; offset++) {
            if (h_uncompressed[i][offset] != h_decompressed[i][offset]) {
                errors++;
            }
        }
        printf("%u\t%u\t%u\t%3.2f\t%u\t%u\n",
               i,
               h_uncompressed_bytes[i],
               h_compressed_bytes[i],
               ratio,
               h_decompressed_bytes[i],
               errors);
    }
    return 0;
};

/*
int main(int argc, char* argv[]) {
    using namespace epigenetic_gol_kernel;
    Simulator simulator(50, 5, 50);
    FitnessGoal goal = FitnessGoal::STILL_LIFE;
    PhenotypeProgram programs[50];
    for (int i = 0; i < 50; i++) {
        programs[i].draw_ops[0].compose_mode = ComposeMode::OR;
        programs[i].draw_ops[0].global_transforms[0].type =
            TransformMode::TILE;
    }
    simulator.evolve(programs, goal, 200);
    return 0;
}
*/
