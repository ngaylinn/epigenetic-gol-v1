#include "simulator.h"

#include <cstdio>
#include <cmath>

#include "gol_simulation.cuh"
#include "selection.cuh"

namespace epigenetic_gol_kernel {
namespace {

__global__ void InitRngsKernel(int population_size, curandState* rngs) {
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index > population_size) return;

    curand_init(42, population_index, 0, &rngs[population_index]);
}

__global__ void NewDefaultInterpretersKernel(Interpreter** interpreters) {
    // New must be called on device for virtual functions to work properly.
    interpreters[threadIdx.x] = new DefaultInterpreter();
}

__global__ void NewMockInterpretersKernel(
        const Frame* frame, Interpreter** interpreters) {
    // New must be called on device for virtual functions to work properly.
    interpreters[threadIdx.x] = new MockInterpreter(frame);
}

__global__ void DeleteInterpretersKernel(Interpreter** interpreters) {
    delete interpreters[threadIdx.x];
}

} // namespace

Simulator::Simulator(
        unsigned int num_species, unsigned int num_trials,
        unsigned int num_organisms)
    : num_species(num_species),
      num_trials(num_trials),
      num_organisms(num_organisms),
      size(num_species * num_trials * num_organisms) {
    // Allocate memory for every aspect of running these simulations. Most data
    // objects are allocated one per individual, except for Interpreters which
    // are one per species.
    cudaMalloc(&interpreters, sizeof(Interpreter*) * num_species);
    cudaMalloc(&rngs, sizeof(curandState) * size);
    cudaMalloc(&genotypes, sizeof(Genotype) * size);
    cudaMalloc(&parent_selections, sizeof(unsigned int) * size);
    cudaMalloc(&mate_selections, sizeof(unsigned int) * size);
    cudaMalloc(&fitness_scores, sizeof(Fitness) * size);
    cudaMalloc(&videos, sizeof(Video) * size);

    // TODO: Actually take Interpreters from Python somehow.
    NewDefaultInterpretersKernel<<<1, num_species>>>(interpreters);

    // Initialize the random number generators.
    reset_state();
}

Simulator::~Simulator() {
    DeleteInterpretersKernel<<<1, num_species>>>(interpreters);
    cudaFree(interpreters);
    cudaFree(rngs);
    cudaFree(genotypes);
    cudaFree(parent_selections);
    cudaFree(mate_selections);
    cudaFree(fitness_scores);
    cudaFree(videos);
}

// ---------------------------------------------------------------------------
// Methods for managing simulation control flow
// ---------------------------------------------------------------------------

void Simulator::populate(Interpreter* h_interpreters) {
    //if (interpreters != nullptr) {
    //    init_interpreters(h_interpreters, num_species, interpreters);
    //}
    randomize_population(size, genotypes, rngs);
}

void Simulator::propagate() {
    select_from_population(size, num_organisms, fitness_scores,
            parent_selections, mate_selections, rngs);

    breed_population(size, parent_selections, mate_selections, genotypes, rngs);
}

void Simulator::simulate(FitnessGoal goal, bool record) {
    simulate_population(
            size, num_species, interpreters, genotypes,
            goal, videos, fitness_scores, record);
}

// ---------------------------------------------------------------------------
// Methods to retrieve simulation results computed by simulate()
// ---------------------------------------------------------------------------

const Fitness* Simulator::get_fitness_scores() const {
    Fitness* result = new Fitness[size]();
    cudaMemcpy(result, fitness_scores, sizeof(Fitness) * size,
            cudaMemcpyDeviceToHost);
    return result;
}

const Video* Simulator::get_videos() const {
    Video* result = new Video[size]();
    cudaMemcpy(result, videos, sizeof(Video) * size,
            cudaMemcpyDeviceToHost);
    return result;
}

const Genotype* Simulator::get_genotypes() const {
    Genotype* result = new Genotype[size]();
    cudaMemcpy(result, genotypes, sizeof(Genotype) * size,
            cudaMemcpyDeviceToHost);
    return result;
}

// ---------------------------------------------------------------------------
// Methods to manage RNG state
// ---------------------------------------------------------------------------

const std::vector<unsigned char> Simulator::get_state() const {
    int state_size = sizeof(curandState) * size;
    std::vector<unsigned char> result(state_size);
    cudaMemcpy(result.data(), rngs, state_size, cudaMemcpyDeviceToHost);
    return result;
}

void Simulator::restore_state(std::vector<unsigned char> state) {
    if (state.size() != size) {
        perror("State object has incorrect size; cannot restore.\n");
    }
    int state_size = sizeof(curandState) * size;
    cudaMemcpy(rngs, state.data(), state_size, cudaMemcpyHostToDevice);
}

void Simulator::reset_state() {
    // Initialize state for generating random numbers.
    InitRngsKernel<<<
        (size + MAX_THREADS - 1) / MAX_THREADS,
        min(size, MAX_THREADS)
    >>>(size, rngs);
}

// ---------------------------------------------------------------------------
// Data injection methods for testing
// ---------------------------------------------------------------------------

void TestSimulator::simulate_phenotype(
        const Frame* h_phenotype, FitnessGoal goal, bool record) {
    Frame* injected_phenotype;
    cudaMalloc(&injected_phenotype, sizeof(Frame));
    cudaMemcpy(injected_phenotype, h_phenotype, sizeof(Frame),
            cudaMemcpyHostToDevice);

    Interpreter** mock_interpreters;
    cudaMalloc(&mock_interpreters, sizeof(Interpreter*) * size);
    NewMockInterpretersKernel<<<1, num_species>>>(
            injected_phenotype, mock_interpreters);

    simulate_population(
            size, num_species, mock_interpreters, nullptr,
            goal, videos, fitness_scores, record);

    DeleteInterpretersKernel<<<1, num_species>>>(mock_interpreters);
    cudaFree(injected_phenotype);
    cudaFree(mock_interpreters);
}

} // namespace epigenetic_gol_kernel
