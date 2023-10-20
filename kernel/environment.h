/*
 * Constants and types representing Game of Life simulations and context for
 * running them, referenced broadly throughout this project.
 */

#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

namespace epigenetic_gol_kernel {

// The length of one side of a square simulated environment.
constexpr unsigned int WORLD_SIZE = 64;
// The number of iterations to run the simulation for.
constexpr unsigned int NUM_STEPS = 100;

// Constants defining how to break down the world into blocks of threads.
constexpr int CELLS_PER_THREAD = 8;
constexpr int REPEATS_PER_ROW = WORLD_SIZE / CELLS_PER_THREAD;
constexpr int THREADS_PER_BLOCK = WORLD_SIZE * REPEATS_PER_ROW;

// The valid states for each cell in the GOL simulation (this could get more
// complex for other kinds of cellular automata).
enum class Cell : unsigned char {
    ALIVE = 0x00,
    DEAD = 0xFF,
    SIZE = 2
};

// The GOL world is just a 2D array of cells.
typedef Cell Frame[WORLD_SIZE][WORLD_SIZE];

// A GOL simulation video is just an array of NUM_STEPS frames.
typedef Frame Video[NUM_STEPS];

// Integer values are used for fitness, larger is better.
typedef unsigned int Fitness;

// Objectives used to evaluate organism fitness.
enum class FitnessGoal {
    NONE,          // No-op fitness function.
    ENTROPY,       // Least compressible final output.
    EXPLODE,       // Fewest live cells -> Most live cells
    LEFT_TO_RIGHT, // Most live cells on left -> Most live cells on right
    RING,          // Most cells arranged in a ring at end.
    STILL_LIFE,    // Most live cells that do not change at end
    SYMMETRY,      // Most live cells at end that mirror (H or V)
    THREE_CYCLE,   // Look for repetition at end with 3-frame cycles
    TWO_CYCLE,     // Look for repetition at end with 2-frame cycles
    SIZE
};

} // namespace epigenetic_gol_kernel

#endif
