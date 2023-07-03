#ifndef __ENVIRONMENT_H__
#define __ENVIRONMENT_H__

/*
 * Constants and types representing Game of Life simulations and conext for
 * running them, referenced broadly throughout this project.
 */

// TODO: Remove this when you're done debugging.
#include <stdio.h>

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

// Which fitness goal to evaluate?
enum class FitnessGoal {
    EXPLODE,
    GLIDERS,
    LEFT_TO_RIGHT,
    STILL_LIFE,
    SYMMETRY,
    THREE_CYCLE,
    TWO_CYCLE,
    SIZE
};

} // namespace epigenetic_gol_kernel

#endif
