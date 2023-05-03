#include "gol_simulation.h"

#include <cub/cub.cuh>

#include "cuda_utils.cuh"
#include "development.cuh"
#include "fitness.cuh"

namespace epigenetic_gol_kernel {
namespace {

constexpr int CELLS_PER_THREAD = 8;
constexpr int REPEATS_PER_ROW = WORLD_SIZE / CELLS_PER_THREAD;
constexpr int THREADS_PER_BLOCK = WORLD_SIZE * REPEATS_PER_ROW;

__device__ __host__ Cell get_next_state(
        const int& curr_row, const int& curr_col, const Frame& last_frame) {
    // Count up neighbors of this Cell that are ALIVE by looking at all the
    // adjacent Cells that are in bounds for this Frame. Bounds checking is
    // done with min / max which is faster than using ifs or ternaries.
    const int prev_row = max(curr_row - 1, 0);
    const int next_row = min(curr_row + 1, WORLD_SIZE - 1);
    const int prev_col = max(curr_col - 1, 0);
    const int next_col = min(curr_col + 1, WORLD_SIZE - 1);
    const int neighbors = (
            (last_frame[prev_row][prev_col] == Cell::ALIVE) +
            (last_frame[prev_row][curr_col] == Cell::ALIVE) +
            (last_frame[prev_row][next_col] == Cell::ALIVE) +
            (last_frame[curr_row][prev_col] == Cell::ALIVE) +
            (last_frame[curr_row][next_col] == Cell::ALIVE) +
            (last_frame[next_row][prev_col] == Cell::ALIVE) +
            (last_frame[next_row][curr_col] == Cell::ALIVE) +
            (last_frame[next_row][next_col] == Cell::ALIVE));

    // Compute the next state for this Cell from the previous state and the
    // number of living neighbors.
    const Cell& last_state = last_frame[curr_row][curr_col];
    return (last_state == Cell::ALIVE && (neighbors == 2 || neighbors == 3) ||
            last_state == Cell::DEAD && neighbors == 3)
        ? Cell::ALIVE : Cell::DEAD;
}

__global__ void GolKernel(
        const FitnessGoal goal,
        const PhenotypeProgram* programs,
        const Genotype* genotypes,
        Video* videos,
        Fitness* fitness_scores,
        bool record) {
    const int& species_index = blockIdx.y;
    const int population_index = blockIdx.y * gridDim.x + blockIdx.x;
    const int row = threadIdx.x / REPEATS_PER_ROW;
    const int col = CELLS_PER_THREAD * (threadIdx.x % REPEATS_PER_ROW);

    const PhenotypeProgram& program = programs[species_index];
    const Genotype& genotype = genotypes[population_index];
    Video& video = videos[population_index];
    Fitness& fitness = fitness_scores[population_index];

    // Calculating the next frame of a Game of Life simulation requires looking
    // at all of a Cell's neighbors in the previous frame. Since it takes many
    // warps to compute a single Frame, this means adjacent warps need to share
    // state for the previous frame (but not the next one). We store one whole
    // Frame in shared memory (in theory you only need enough to represent the
    // seams between warps, but that scheme would require extra work) and just
    // the cells for the next frame being computed by this thread in registers.
    __shared__ Frame last_frame;
    Cell curr_frame[CELLS_PER_THREAD];

    // Each frame may contribute to overall fitness. This function doesn't
    // always record all those frames, so we compute fitness incrementally
    // frame by frame, storing partial work here.
    PartialFitness partial_fitness[CELLS_PER_THREAD];

    // Interpret this organism's genotype to generate the phenotype, which is
    // the first frame of the simulation.
    for (int i = 0; i < CELLS_PER_THREAD; i++) {
        // TODO: Pass last_frame as a workspace for the stack? It only has
        // depth one, but that's enough for most simple compositions.
        make_phenotype(program, genotype, row, col+i, curr_frame[i]);
    }

    // Make sure this frame is finished before looking at it.
    __syncthreads();

    // Run the simulated lifetime...
    for (int step = 0; step < NUM_STEPS; step++) {
        // Copy the most recently computed frame data into shared memory and
        // wait for it to finish before calling get_next_state below. Since
        // each thread works on CELLS_PER_THREAD contiguous Cells, we can do
        // this with a single memcpy instead of a loop. Doing this well before
        // it's needed seems to help hide the memory access latency.
        memcpy(&last_frame[row][col], curr_frame, sizeof(curr_frame));
        __syncthreads();

        // If recording, save a copy of each frame to global memory.
        if (record) {
            memcpy(&video[step][row][col], curr_frame, sizeof(curr_frame));
        }

        // Compute the fitness contribution of each frame as we go along.
        for (int i = 0; i < CELLS_PER_THREAD; i++) {
            update_fitness(
                    step, row, col+i, curr_frame[i], goal, partial_fitness[i]);
        }

        // If we've already computed, evaluated, and saved the last frame, then
        // stop here before computing another one.
        if (step == NUM_STEPS - 1) break;

        // Compute the next frame from the previous one.
        for (int i = 0; i < CELLS_PER_THREAD; i++) {
            curr_frame[i] = get_next_state(row, col+i, last_frame);
        }
        __syncthreads();
    }

    // Finalize all the fitness scores.
    for (int i = 0; i < CELLS_PER_THREAD; i++) {
        finalize_fitness(goal, partial_fitness[i]);
    }

    // Sum fitness contributions from each thread.
    auto reduce = cub::BlockReduce<Fitness, THREADS_PER_BLOCK>();
    int sum = reduce.Sum((Fitness(&)[CELLS_PER_THREAD]) partial_fitness);

    // Save the final result to global memory to return to the host.
    if (threadIdx.x == 0) {
        fitness = sum;
    }
}

} // namespace

void simulate_population(
        const unsigned int population_size,
        const unsigned int num_species,
        const FitnessGoal& goal,
        const PhenotypeProgram* programs,
        const Genotype* genotypes,
        Video* videos,
        Fitness* fitness_scores,
        bool record) {
    GolKernel<<<
        { population_size / num_species, num_species },
        THREADS_PER_BLOCK
    >>>(goal, programs, genotypes, videos, fitness_scores, record);
    CUDA_CHECK_ERROR();
}

Video* simulate_phenotype(const Frame& phenotype) {
    Video* video = (Video*) new Video;
    // Fill in the first frame of the Video from the phenotype
    memcpy(video, &phenotype, sizeof(Frame));
    // Compute the remaining frames from the first one.
    for (int step = 1; step < NUM_STEPS; step++) {
        for (int row = 0; row < WORLD_SIZE; row++) {
            for (int col = 0; col < WORLD_SIZE; col++) {
                (*video)[step][row][col] = 
                    get_next_state(row, col, (*video)[step-1]);
            }
        }
    }
    return video;
}

namespace {

__global__ void MakePhenotypeKernel(
        const PhenotypeProgram& program,
        const Genotype* genotypes,
        Frame* phenotypes) {
    const int population_index = blockIdx.x;
    const int row = threadIdx.x / REPEATS_PER_ROW;
    const int col = CELLS_PER_THREAD * (threadIdx.x % REPEATS_PER_ROW);

    const Genotype& genotype = genotypes[population_index];
    Frame& phenotype = phenotypes[population_index];

    for (int i = 0; i < CELLS_PER_THREAD; i++) {
        make_phenotype(program, genotype, row, col+i,
                (Cell&) phenotype[row][col+i]);
    }
}

} // namespace

const Frame* render_phenotype(
        const PhenotypeProgram& h_program,
        const Genotype* h_genotype) {
    DeviceData<PhenotypeProgram> program(&h_program);
    DeviceData<Genotype> genotype;
    // Either use the given Genotype, or use an empty one as a default.
    if (h_genotype) {
        genotype.copy_from_host(h_genotype);
    } else {
        CUDA_CALL(cudaMemset(genotype, 0, sizeof(Genotype)));
    }
    DeviceData<Frame> phenotype;

    MakePhenotypeKernel<<<
        1, THREADS_PER_BLOCK
    >>>(program, genotype, phenotype);
    return phenotype.copy_to_host();
}
} // namespace epigenetic_gol_kernel
