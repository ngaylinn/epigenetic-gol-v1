#include "fitness.cuh"

#include <cstdint>

#include <cub/cub.cuh>
#include <nvcomp.h>
#include <nvcomp/gdeflate.h>

#include "cuda_utils.cuh"
#include "environment.h"

namespace epigenetic_gol_kernel {

// ---------------------------------------------------------------------------
// FitnessObserver Implementation.
// ---------------------------------------------------------------------------

template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::observe(
        const int& step, const int& row, const int& col,
        const Cell local[CELLS_PER_THREAD], const Frame& global) {
    for (int i = 0; i < CELLS_PER_THREAD; i++) {
        // Only some FitnessGoals require a global view of the GOL simulation,
        // which is a bit less efficient than using a local view. The compiler
        // can optimize away the switch statement and directly call the correct
        // method for each goal.
        switch(GOAL) {
            case FitnessGoal::EXPLODE:
            case FitnessGoal::LEFT_TO_RIGHT:
            case FitnessGoal::RING:
            case FitnessGoal::STILL_LIFE:
            case FitnessGoal::THREE_CYCLE:
            case FitnessGoal::TWO_CYCLE:
                // Observe this Cell and store incremental fitness data in
                // scratch_a[i] and scratch_b[i]. The meaning of these two values
                // is goal-specific.
                update(step, row, col+i, local[i], scratch_a[i], scratch_b[i]);
                break;

            case FitnessGoal::SYMMETRY:
                // Observe this Cell and store incremental fitness data in
                // scratch_a[i] and scratch_b[i]. The meaning of these two values
                // is goal-specific.
                update(step, row, col+i, global, scratch_a[i], scratch_b[i]);
                break;

            default:
                break;
        }
    }
}

template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::reduce(Fitness* result) {
    // Add up the values of scratch_a and scratch_b across all threads
    // (ie, the full GOL world).
    auto reduce = cub::BlockReduce<uint32_t, THREADS_PER_BLOCK>();
    uint32_t sum_a = reduce.Sum(scratch_a);
    // Needed because both calls to Sum share the same workspace memory.
    __syncthreads();
    uint32_t sum_b = reduce.Sum(scratch_b);

    // CUB returns the final reduction value in thread 0. Use a goal-specific
    // function to translate the two values in sum_a and sum_b into a single
    // fitness score.
    if (threadIdx.x == 0) {
        return finalize(sum_a, sum_b, result);
    }
}

// There's a version of FitnessObserver for every FitnessGoal, and each one
// must implement one of the two update methods. Empty implementations are
// provided so the compiler won't complain about a missing definition for
// whichever version goes unused. The finalize method also has a default
// implementation, mostly for supporting the NONE and ENTROPY FitnessGoals,
// which don't have their own implementation.
template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::update(
        const int& step, const int& row, const int& col, const Frame& frame,
        uint32_t& scratch_a, uint32_t& scratch_b) {}
template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& scratch_a, uint32_t& scratch_b) {}
template<FitnessGoal GOAL>
__device__ void FitnessObserver<GOAL>::finalize(
        const uint32_t& alive_at_start, const uint32_t& alive_at_end,
        Fitness* result) {}

// ---------------------------------------------------------------------------
// Explode
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::EXPLODE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& alive_at_start, uint32_t& alive_at_end) {
    if (step > 0 && step < NUM_STEPS - 1) return;

    if (step == 0) {
        alive_at_start = (cell == Cell::ALIVE);
    } else { // step == NUM_STEPS - 1
        alive_at_end += (cell == Cell::ALIVE);
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::EXPLODE>::finalize(
        const uint32_t& alive_at_start, const uint32_t& alive_at_end,
        Fitness* result) {
    *result = (100 * alive_at_end) / (1 + alive_at_start);
}


// ---------------------------------------------------------------------------
// LeftToRight
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& on_target_first_frame, uint32_t& on_target_last_frame) {
    if (step > 0 && step < NUM_STEPS - 1) return;

    const bool alive = (cell == Cell::ALIVE);

    // On the first step, a cell is "on target" if it is ALIVE on the left or
    // DEAD on the right. The opposite is true for the last step.
    if (step == 0) {
        on_target_first_frame = alive == (col < WORLD_SIZE / 2);
    } else if (step == NUM_STEPS - 1) {
        on_target_last_frame = alive == (col >= WORLD_SIZE / 2);
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>::finalize(
        const uint32_t& on_target_first_frame,
        const uint32_t& on_target_last_frame,
        Fitness* result) {
    // Look for simulations with high on-target values for the first and last
    // steps. Weight the last step higher than the first, since it's much
    // easier to craft a good starting phenotype than to have the simulation
    // ultimately produce a good last step.
    *result = on_target_first_frame + 4 * on_target_last_frame;
}


// ---------------------------------------------------------------------------
// StillLife
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::STILL_LIFE>::update(
        const int& step, const int& row, const int& col,
        const Cell& cell, uint32_t& static_cells, uint32_t& live_cells) {
    if (step < NUM_STEPS - 2) return;

    const bool alive = (cell == Cell::ALIVE);

    if (step == NUM_STEPS - 2) {
        static_cells = alive;
    } else {  // step == NUM_STEPS - 1
        static_cells = static_cells && alive;
        live_cells = alive;
    }
}

template<>
__device__ void FitnessObserver<FitnessGoal::STILL_LIFE>::finalize(
        const uint32_t& static_cells, const uint32_t& live_cells,
        Fitness* result) {
    *result = (100 * static_cells) / (1 + live_cells - static_cells);
}


// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::SYMMETRY>::update(
        const int& step, const int& row, const int& col,
        const Frame& frame, uint32_t& symmetries, uint32_t& assymmetries) {
    if (step < NUM_STEPS - 1) return;

    const bool alive = frame[row][col] == Cell::ALIVE;
    const bool v_mirror = frame[row][WORLD_SIZE - 1 - col] == Cell::ALIVE;
    const bool h_mirror = frame[WORLD_SIZE - 1 - row][col] == Cell::ALIVE;
    symmetries = int(alive && h_mirror) + int(alive && v_mirror);
    assymmetries = int(alive && !h_mirror) + int(alive && !v_mirror);
}

template<>
__device__ void FitnessObserver<FitnessGoal::SYMMETRY>::finalize(
        const uint32_t& symmetries, const uint32_t& assymmetries,
        Fitness* result) {
    *result = (100 * symmetries) / (1 + assymmetries);
}


// ---------------------------------------------------------------------------
// ThreeCycle
// ---------------------------------------------------------------------------

namespace {
template<int CYCLE_LENGTH>
__device__ void update_cycle(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    // How many times must the cycle repeat in order to count it?
    constexpr int NUM_ITERATIONS = 4;
    // A bitmask to capture the last CYCLE_LENGTH bits.
    constexpr int MASK = (2 << (CYCLE_LENGTH - 1)) - 1;

    // Only consider the last few steps of the simulation, just enough to
    // capture the desired number of cycles.
    if (step < NUM_STEPS - CYCLE_LENGTH * NUM_ITERATIONS) return;

    // Record a history of this cell's state, one bit per simulation step.
    history = history << 1 | (cell == Cell::ALIVE);

    // Only on the last step, review the history to count cycling Cells.
    if (step < NUM_STEPS - 1) return;

    // Capture the pattern found in the last N steps of the simulation, then
    // look at previous iterations of the cycle to see if it repeats.
    const int last_cycle = history & MASK;
    cycling = true;
    bool not_cycling = false;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        // Grab the last N states of this cycle from the history variable,
        // then bitshift to set up the next N for the next loop iteration.
        const int this_cycle = history & MASK;
        history >>= CYCLE_LENGTH;

        const bool always_off = (this_cycle == 0);
        const bool always_on = (this_cycle == MASK);
        const bool repeating = (this_cycle == last_cycle);
        // True iff this cell is in a repeating, non-static pattern.
        cycling &= !always_off && !always_on && repeating;
        // True for any cell that was on at some point but was not repeating.
        not_cycling |= !(always_off || repeating);
    }
    // Reuse the history variable to track not_cycling.
    history = not_cycling;
}
} // namespace

template<>
__device__ void FitnessObserver<FitnessGoal::THREE_CYCLE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    update_cycle<3>(step, row, col, cell, history, cycling);
}

template<>
__device__ void FitnessObserver<FitnessGoal::THREE_CYCLE>::finalize(
        const uint32_t& not_cycling, const uint32_t& cycling, Fitness* result) {
    // Prefer simulations with more Cells cycling, and with relatively little
    // "debris" that's not participating.
    *result = (cycling * cycling) / (1 + not_cycling);
}


// ---------------------------------------------------------------------------
// TwoCycle
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& history, uint32_t& cycling) {
    update_cycle<2>(step, row, col, cell, history, cycling);
}

template<>
__device__ void FitnessObserver<FitnessGoal::TWO_CYCLE>::finalize(
        const uint32_t& not_cycling, const uint32_t& cycling, Fitness* result) {
    // Prefer simulations with more Cells cycling, and with relatively little
    // "debris" that's not participating.
    *result = (cycling * cycling) / (1 + not_cycling);
}


// ---------------------------------------------------------------------------
// Ring
// ---------------------------------------------------------------------------

template<>
__device__ void FitnessObserver<FitnessGoal::RING>::update(
        const int& step, const int& row, const int& col, const Cell& cell,
        uint32_t& on_target, uint32_t& off_target) {
    if (step < NUM_STEPS - 1) return;

    // These constants describe a ring of live Cells in the center of the GOL
    // simulation, with dead space in the middle and around the ring.
    constexpr int CENTER = WORLD_SIZE / 2;
    constexpr int INNER_RADIUS = CENTER / 4;
    constexpr int OUTER_RADIUS = 3 * CENTER / 4;

    // Calculate whether this Cell is within the ring or outside it. Note that
    // we compare squared distance values rather than the actual distance
    // (which requires calculating square roots, which is slow).
    int distance_squared = (
        (CENTER - row) * (CENTER - row) +
        (CENTER - col) * (CENTER - col));
    bool within_target = (
        (distance_squared > INNER_RADIUS * INNER_RADIUS) &&
        (distance_squared <= OUTER_RADIUS * OUTER_RADIUS));

    on_target = within_target == (cell == Cell::ALIVE);
    off_target = !on_target;
}

template<>
__device__ void FitnessObserver<FitnessGoal::RING>::finalize(
        const uint32_t& on_target, const uint32_t& off_target, Fitness* result) {
    *result = (100 * on_target) / (1 + off_target);
}


// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

// The Entropy FitnessGoal requires invoking the nvcomp library to find the
// compression rates for a batch of simulation Videos. That API call can't be
// called from the GPU, which means the Entropy FitnessGoal can't be computed
// incrementally while the simulation is being run. Instead, it's computed at
// the end, with access to the full simulation Video. Unfortunately, this is
// much slower than the other goals (about 12x).

// This GPU kernel finalizes the fitness computation setup by compute_entropy.
__global__ void EntropyFitnessKernel(
        const int population_size,
        Video* videos,
        size_t* compressed_bytes,
        Fitness* fitness_scores) {
    const int population_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (population_index >= population_size) return;

    // Grab the compressed byte size of the first and last Vrame from the GOL
    // simulation of a single organism.
    const size_t first_frame_size = compressed_bytes[2 * population_index];
    const size_t last_frame_size = compressed_bytes[2 * population_index + 1];

    // Prefer simulations that are less compressible, both in their first Frame
    // and in their last Frame.
    fitness_scores[population_index] = last_frame_size * first_frame_size;
}

void compute_entropy(
        const int population_size, Video* d_videos, Fitness* d_fitness_scores) {
    // Parameters for compression. For each organism, consider both the first
    // and the last Frame from their simulation (two samples per organism).
    const unsigned int num_samples = 2 * population_size;
    const nvcompBatchedGdeflateOpts_t options = { 1 };

    // Compute space needed by the compression algorithm.
    size_t max_compressed_bytes;
    size_t temp_bytes;
    nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
        sizeof(Frame), options, &max_compressed_bytes);
    nvcompBatchedGdeflateCompressGetTempSize(
        num_samples, sizeof(Frame), options, &temp_bytes);

    // Allocate device-side memory to hold the compressed Frames.
    DeviceData<uint8_t> d_compressed(max_compressed_bytes * num_samples);

    // Setup host-side data structures for pointers and byte counts. The
    // *ptrs arrays point to each item in the data blocks for compressed and
    // uncompressed Frames. The *bytes arrays is a parallel array indicating
    // the size in bytes for each item to compress.
    void* h_uncompressed_ptrs[num_samples];
    void* h_compressed_ptrs[num_samples];
    size_t h_uncompressed_bytes[num_samples];
    size_t h_compressed_bytes[num_samples];
    for (int i = 0; i < population_size; i++) {
        // Convert the single population index into indexes for the two samples
        // for each individual in the population.
        int sample1 = 2 * i;
        int sample2 = 2 * i + 1;
        // Calculate pointers into the input and output data structures. For
        // the input, we point to the first and last Frame of each simulation
        // Video. For output, we just point at contiguous spaces in the memory
        // block.
        h_uncompressed_ptrs[sample1] = &(d_videos[i][0]);
        h_uncompressed_ptrs[sample2] = &(d_videos[i][NUM_STEPS - 1]);
        h_compressed_ptrs[sample1] = &(d_compressed[sample1 * max_compressed_bytes]);
        h_compressed_ptrs[sample2] = &(d_compressed[sample2 * max_compressed_bytes]);
        // All the input chunks have the same size (one Frame from the Video).
        h_uncompressed_bytes[sample1] = sizeof(Frame);
        h_uncompressed_bytes[sample2] = sizeof(Frame);
    }

    // Setup device-side mirrors for the host-side data initialized above.
    DeviceData<void*> d_uncompressed_ptrs(num_samples, h_uncompressed_ptrs);
    DeviceData<void*> d_compressed_ptrs(num_samples, h_compressed_ptrs);
    DeviceData<size_t> d_uncompressed_bytes(num_samples, h_uncompressed_bytes);
    DeviceData<size_t> d_compressed_bytes(num_samples);

    // Compress samples from every organism of every species in a single batch.
    DeviceData<unsigned char> d_temp(temp_bytes);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    nvcompStatus_t status = nvcompBatchedGdeflateCompressAsync(
            d_uncompressed_ptrs,
            d_uncompressed_bytes,
            sizeof(Frame),
            num_samples,
            d_temp,
            temp_bytes,
            d_compressed_ptrs,
            d_compressed_bytes,
            options,
            stream);
    cudaDeviceSynchronize();

    // Ignore the compressed Video data, and just look at the size in bytes of
    // each compressed Video to compute the fitness scores. This is run on the
    // GPU (one thread per organism) because that's where the data is.
    unsigned int organisms_per_block = min(MAX_THREADS, population_size);
    unsigned int num_blocks =
        (population_size + organisms_per_block - 1) / organisms_per_block;
    EntropyFitnessKernel<<<
        num_blocks, organisms_per_block
    >>>(population_size, d_videos, d_compressed_bytes, d_fitness_scores);
}

// ---------------------------------------------------------------------------

// Make sure we actually instantiate a version of the class for every goal.
template class FitnessObserver<FitnessGoal::NONE>;
template class FitnessObserver<FitnessGoal::EXPLODE>;
template class FitnessObserver<FitnessGoal::LEFT_TO_RIGHT>;
template class FitnessObserver<FitnessGoal::RING>;
template class FitnessObserver<FitnessGoal::STILL_LIFE>;
template class FitnessObserver<FitnessGoal::SYMMETRY>;
template class FitnessObserver<FitnessGoal::THREE_CYCLE>;
template class FitnessObserver<FitnessGoal::TWO_CYCLE>;

} // namespace epigenetic_gol_kernel
