#include "simulator.h"

#include <utility>
#include <vector>

#include "cuda_utils.cuh"
#include "gol_simulation.h"
#include "reproduction.h"
#include "selection.h"

namespace epigenetic_gol_kernel {
namespace {

__global__ void InitRngsKernel(int population_size, curandState* rngs) {
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index >= population_size) return;

    curand_init(42, population_index, 0, &rngs[population_index]);
}

// For convenience, set up a PhenotypeProgram that's hard-coded instead of
// evolved. This is a simple example, based on the "Tile" configuration in the
// prototype of this project.
__global__ void InitDefaultPrograms(PhenotypeProgram* programs) {
    PhenotypeProgram& program = programs[threadIdx.x];
    Operation& translate = program.ops[0];
    translate.type = OperationType::TRANSLATE;
    translate.args[0].gene_index = 0;
    translate.args[0].gene_bias.scalar = 0;
    translate.args[0].bias_mode = BiasMode::FIXED_VALUE;
    translate.args[1].gene_index = 0;
    translate.args[1].gene_bias.scalar = 0;
    translate.args[1].bias_mode = BiasMode::FIXED_VALUE;
    translate.next_op_index = 1;

    Operation& repeat = program.ops[1];
    repeat.type = OperationType::ARRAY_2D;
    repeat.args[0].gene_index = 0;
    repeat.args[0].gene_bias.scalar = 8 * (UINT_MAX / WORLD_SIZE);
    repeat.args[0].bias_mode = BiasMode::FIXED_VALUE;
    repeat.args[1].gene_index = 0;
    repeat.args[1].gene_bias.scalar = 8 * (UINT_MAX / WORLD_SIZE);
    repeat.args[1].bias_mode = BiasMode::FIXED_VALUE;
    repeat.next_op_index = 2;

    Operation& draw = program.ops[2];
    draw.type = OperationType::DRAW;
    draw.args[0].gene_index = 0;
    draw.args[0].bias_mode = BiasMode::NONE;
    draw.next_op_index = STOP_INDEX;
}

} // namespace

// TODO: Consider running simulations in parallel with breeding
// PhenotypePrograms in python, if that makes sense.
Simulator::Simulator(
        unsigned int num_species, unsigned int num_trials,
        unsigned int num_organisms)
    : num_species(num_species),
      num_trials(num_trials),
      num_organisms(num_organisms),
      size(num_species * num_trials * num_organisms) {

    // Allocate GPU device memory for every aspect of running these
    // simulations. Most data objects are allocated one per individual, except
    // for PhenotypePrograms which are one per species.
    CUDA_CALL(cudaMalloc(&programs, sizeof(PhenotypeProgram) * num_species));
    CUDA_CALL(cudaMalloc(&rngs, sizeof(curandState) * size));
    CUDA_CALL(cudaMalloc(&genotype_buffer_0, sizeof(Genotype) * size));
    CUDA_CALL(cudaMalloc(&genotype_buffer_1, sizeof(Genotype) * size));
    CUDA_CALL(cudaMalloc(&parent_selections, sizeof(unsigned int) * size));
    CUDA_CALL(cudaMalloc(&mate_selections, sizeof(unsigned int) * size));
    CUDA_CALL(cudaMalloc(&fitness_scores, sizeof(Fitness) * size));
    CUDA_CALL(cudaMalloc(&videos, sizeof(Video) * size));

    // Allocate host memory for transferring data to / from the GPU.
    CUDA_CALL(cudaMallocHost(
                &h_programs, sizeof(PhenotypeProgram) * num_species));
    CUDA_CALL(cudaMallocHost(&h_fitness_scores, sizeof(Fitness) * size));
    CUDA_CALL(cudaMallocHost(&h_videos, sizeof(Video) * size));
    CUDA_CALL(cudaMallocHost(&h_genotypes, sizeof(Genotype) * size));

    // Use a double buffer for genotypes, switching on each generation.
    curr_gen_genotypes = genotype_buffer_0;
    next_gen_genotypes = genotype_buffer_1;

    InitDefaultPrograms<<<1, num_species>>>(programs);

    // Initialize the random number generators.
    reset_state();
}

Simulator::~Simulator() {
    CUDA_CALL(cudaFree(programs));
    CUDA_CALL(cudaFree(rngs));
    CUDA_CALL(cudaFree(genotype_buffer_0));
    CUDA_CALL(cudaFree(genotype_buffer_1));
    CUDA_CALL(cudaFree(parent_selections));
    CUDA_CALL(cudaFree(mate_selections));
    CUDA_CALL(cudaFree(fitness_scores));
    CUDA_CALL(cudaFree(videos));

    CUDA_CALL(cudaFreeHost(h_programs));
    CUDA_CALL(cudaFreeHost(h_fitness_scores));
    CUDA_CALL(cudaFreeHost(h_videos));
    CUDA_CALL(cudaFreeHost(h_genotypes));
}


// ---------------------------------------------------------------------------
// Methods for managing simulation control flow
// ---------------------------------------------------------------------------

// TODO: Actually pass PhenotypePrograms from Python.
void Simulator::populate(PhenotypeProgram* h_programs) {
    randomize_population(size, num_organisms, curr_gen_genotypes, rngs);
}

void Simulator::propagate() {
    select_from_population(size, num_organisms, fitness_scores,
            parent_selections, mate_selections, rngs);

    breed_population(
            size, num_organisms, parent_selections, mate_selections,
            curr_gen_genotypes, next_gen_genotypes, rngs);

    // After generationg next_gen_genotypes from curr_gen_genotypes, swap the
    // two pointers so that the the new data is "current" and the old data is
    // available for computing the next generation.
    std::swap(curr_gen_genotypes, next_gen_genotypes);
}

void Simulator::simulate(FitnessGoal goal, bool record) {
    simulate_population(
            size, num_species, goal, programs, curr_gen_genotypes,
            videos, fitness_scores, record);
}


// ---------------------------------------------------------------------------
// Methods to retrieve simulation results computed by simulate()
// ---------------------------------------------------------------------------

const Fitness* Simulator::get_fitness_scores() const {
    CUDA_CALL(cudaMemcpy(h_fitness_scores, fitness_scores,
                sizeof(Fitness) * size, cudaMemcpyDeviceToHost));
    return h_fitness_scores;
}

// TODO: Consider optimizing this to only copy the videos you want.
const Video* Simulator::get_videos() const {
    CUDA_CALL(cudaMemcpy(h_videos, videos,
                sizeof(Video) * size, cudaMemcpyDeviceToHost));
    return h_videos;
}

const Genotype* Simulator::get_genotypes() const {
    CUDA_CALL(cudaMemcpy(h_genotypes, curr_gen_genotypes,
                sizeof(Genotype) * size, cudaMemcpyDeviceToHost));
    return h_genotypes;
}


// ---------------------------------------------------------------------------
// Methods to manage RNG state
// ---------------------------------------------------------------------------

const std::vector<unsigned char> Simulator::get_state() const {
    int state_size = sizeof(curandState) * size;
    std::vector<unsigned char> result(state_size);
    CUDA_CALL(cudaMemcpy(result.data(), rngs, state_size,
                cudaMemcpyDeviceToHost));
    return result;
}

void Simulator::restore_state(std::vector<unsigned char> state) {
    if (state.size() != size) {
        perror("State object has incorrect size; cannot restore.\n");
    }
    int state_size = sizeof(curandState) * size;
    CUDA_CALL(cudaMemcpy(rngs, state.data(), state_size,
                cudaMemcpyHostToDevice));
}

void Simulator::reset_state() {
    // Initialize state for generating random numbers.
    InitRngsKernel<<<
        (size + MAX_THREADS - 1) / MAX_THREADS,
        min(size, MAX_THREADS)
    >>>(size, rngs);
    CUDA_CHECK_ERROR();
}


// ---------------------------------------------------------------------------
// Methods to inject data for testing
// ---------------------------------------------------------------------------

const Genotype* Simulator::breed_genotypes(
        const Genotype* genotype_data,
        std::vector<unsigned int> h_parent_selections,
        std::vector<unsigned int> h_mate_selections) {
    CUDA_CALL(cudaMemcpy(curr_gen_genotypes, genotype_data,
                sizeof(Genotype) * size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(parent_selections, h_parent_selections.data(),
                sizeof(unsigned int) * size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(mate_selections, h_mate_selections.data(),
                sizeof(unsigned int) * size, cudaMemcpyHostToDevice));

    breed_population(
            size, num_organisms, parent_selections, mate_selections,
            curr_gen_genotypes, next_gen_genotypes, rngs);
    std::swap(curr_gen_genotypes, next_gen_genotypes);

    return get_genotypes();
}

} // namespace epigenetic_gol_kernel


// ---------------------------------------------------------------------------
// A pure C++ demo for use with NVidia's profiling and debug tools.
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    using namespace epigenetic_gol_kernel;
    Simulator simulator(32, 5, 32);
    FitnessGoal goal = FitnessGoal::STILL_LIFE;
    simulator.populate();
    for (int i = 0; i < 199; i++) {
        simulator.simulate(goal);
        simulator.propagate();
        // Include this GPU->host data transfer that the real project requires.
        simulator.get_fitness_scores();
    }
    simulator.simulate(goal, true);
    // Include this GPU->host data transfer that the real project requires.
    simulator.get_videos();
    return 0;
}
