#ifndef __DEVELOPMENT_H__
#define __DEVELOPMENT_H__

/*
 * Primitive functions for building a phenotype.
 *
 * In this project, developing a phenotype is achieved by visiting every Cell
 * in a Frame, then applying a sequence of functions to compute the value for
 * that Cell based on evolved data in a Genotype. This module is for collecting
 * all the different functions that the Operations in interpreter.cuh can run.
 */

#include "environment.h"

namespace epigenetic_gol_kernel {

// Forward declared from interpreter.cuh to avoid a circular dependency.
class BoundArguments;

// An exhaustive list of Operations. When adding a new apply func, make sure to
// add a new enum value here.
enum class OperationType {
    ARRAY_1D,  // Repeat phenotype pattern in a line
    ARRAY_2D,  // Repeat phenotype pattern in a grid
    COPY,      // Copy the phenotype pattern once, with some offset
    DRAW,      // Draw a Stamp onto the phenotype
    TRANSLATE, // Shift phenotype pattern by some offset
    SIZE
};

// Apply an Operation function, potentially setting the value of cell to modify
// the phenotype, or changing the values of row and col to modify the indexing
// scheme for other Operations.
__device__ void apply_operation(
        OperationType type, const BoundArguments& args,
        int& row, int& col, Cell& cell);

} // namespace epigenetic_gol_kernel

#endif
