// TODO: Why is environment included here? Either move it below or add a comment.
#include "environment.h"
#include "gol_simulation.h"

// TODO: Unused
#include <cub/cub.cuh>

#include "cuda_utils.cuh"
#include "development.cuh"
#include "fitness.cuh"

namespace epigenetic_gol_kernel {

namespace {

// A parallel implementation of the GOL state transition function
__device__ __host__ Cell get_next_state(
        const int& curr_row, const int& curr_col, const Frame& last_frame) {
    // Count up neighbors of this Cell that are ALIVE by looking at all the
    // adjacent Cells that are in bounds for this Frame. Bounds checking is
    // done with min / max which is faster than using ifs or ternaries. This
    // produces characteristic quirky behavior at the edges of the board. 
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

// This kernel is templatized so that some logic can be resolved before the
// kernel is launched instead of having if / switch statements that happen on
// every iteration of the innermost loop (which has high latency costs).
template<FitnessGoal GOAL, bool RECORD>
__global__ void GolKernel(
        const PhenotypeProgram* programs,
        const Genotype* genotypes,
        Fitness* fitness_scores,
        Video* videos=nullptr) {
    const int& species_index = blockIdx.y;
    const int population_index = blockIdx.y * gridDim.x + blockIdx.x;
    const int row = threadIdx.x / REPEATS_PER_ROW;
    const int col = CELLS_PER_THREAD * (threadIdx.x % REPEATS_PER_ROW);

    const PhenotypeProgram& program = programs[species_index];
    const Genotype& genotype = genotypes[population_index];
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

    FitnessObserver<GOAL> fitness_observer;

    // Interpret this organism's genotype to generate the phenotype, which is
    // the first frame of the simulation.
    for (int i = 0; i < CELLS_PER_THREAD; i++) {
        // TODO: This operation can be quite expensive. To further optimize
        // this, try "compiling" the phenotype program once per species trial.
        // This could reduce the overhead of interpreting the program and
        // deciding which operations to call on every iteration.
        make_phenotype(program, genotype, row, col+i, curr_frame[i]);
    }

    // Make sure this frame is finished before looking at it.
    __syncthreads();

    // Run the simulated lifetime...
    for (int step = 0; step < NUM_STEPS; step++) {
        // Copy the most recently computed frame data into shared memory and
        // wait for it to finish before calling get_next_state below. Since
        // each thread works on CELLS_PER_THREAD contiguous Cells, we can do
        // this with a single memcpy instead of a loop.
        memcpy(&last_frame[row][col], curr_frame, sizeof(curr_frame));
        __syncthreads();

        // Record videos of all simulations, but only if requested because
        // it's expensive to do that. This check will be optimized away by
        // the compiler.
        if (RECORD) {
            memcpy(&videos[population_index][step][row][col],
                    curr_frame, sizeof(curr_frame));
        }

        fitness_observer.observe(step, row, col, curr_frame, last_frame);

        // If we've already computed, evaluated, and saved the last frame, then
        // stop here before computing another one.
        if (step == NUM_STEPS - 1) break;

        // Compute the next frame from the previous one.
        for (int i = 0; i < CELLS_PER_THREAD; i++) {
            curr_frame[i] = get_next_state(row, col+i, last_frame);
        }
        __syncthreads();
    }

    fitness_observer.reduce(&fitness);
}

} // namespace

// RECORD is passed as a template option here mostly to simplify launching the
// GolKernel. If it was passed as a regular argument, it would double the
// number of cases in the switch statement.
template<bool RECORD>
void simulate_population(
        const unsigned int population_size,
        const unsigned int num_species,
        const FitnessGoal& goal,
        const PhenotypeProgram* programs,
        const Genotype* genotypes,
        Fitness* fitness_scores,
        Video* videos) {
    dim3 grid = { population_size / num_species, num_species };
    switch (goal) {
        case FitnessGoal::EXPLODE:
            GolKernel<FitnessGoal::EXPLODE, RECORD><<<
                grid, THREADS_PER_BLOCK
            >>>(programs, genotypes, fitness_scores, videos);
            break;
        case FitnessGoal::GLIDERS:
            GolKernel<FitnessGoal::GLIDERS, RECORD><<<
                grid, THREADS_PER_BLOCK
            >>>(programs, genotypes, fitness_scores, videos);
            break;
        case FitnessGoal::LEFT_TO_RIGHT:
            GolKernel<FitnessGoal::LEFT_TO_RIGHT, RECORD><<<
                grid, THREADS_PER_BLOCK
            >>>(programs, genotypes, fitness_scores, videos);
            break;
        case FitnessGoal::STILL_LIFE:
            GolKernel<FitnessGoal::STILL_LIFE, RECORD><<<
                grid, THREADS_PER_BLOCK
            >>>(programs, genotypes, fitness_scores, videos);
            break;
        case FitnessGoal::SYMMETRY:
            GolKernel<FitnessGoal::SYMMETRY, RECORD><<<
                grid, THREADS_PER_BLOCK
            >>>(programs, genotypes, fitness_scores, videos);
            break;
        case FitnessGoal::THREE_CYCLE:
            GolKernel<FitnessGoal::THREE_CYCLE, RECORD><<<
                grid, THREADS_PER_BLOCK
            >>>(programs, genotypes, fitness_scores, videos);
            break;
        case FitnessGoal::TWO_CYCLE:
            GolKernel<FitnessGoal::TWO_CYCLE, RECORD><<<
                grid, THREADS_PER_BLOCK
            >>>(programs, genotypes, fitness_scores, videos);
            break;
        default:
            return;
    }
    CUDA_CHECK_ERROR();
}

// Instantiate versions of this function with and without recording.
template void simulate_population<true>(
        const unsigned int population_size, const unsigned int num_species,
        const FitnessGoal& goal, const PhenotypeProgram* programs,
        const Genotype* genotypes, Fitness* fitness_scores, Video* videos);

template void simulate_population<false>(
        const unsigned int population_size, const unsigned int num_species,
        const FitnessGoal& goal, const PhenotypeProgram* programs,
        const Genotype* genotypes, Fitness* fitness_scores, Video* videos);

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

Video* simulate_organism(
        const PhenotypeProgram& h_program,
        const Genotype& h_genotype) {
    const Frame* frame = render_phenotype(h_program, &h_genotype);
    Video* video = simulate_phenotype(*frame);
    delete[] frame;
    return video;
}

} // namespace epigenetic_gol_kernel
