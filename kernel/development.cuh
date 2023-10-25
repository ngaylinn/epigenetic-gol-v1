/*
 * Primitive functions for building a phenotype.
 *
 * In this project, developing a phenotype is achieved by visiting every Cell
 * in a Frame, then applying a sequence of functions to compute the value for
 * that Cell based on evolved data in a Genotype. This module is for collecting
 * all the different functions that the Operations in interpreter.cuh can run.
 */

#ifndef __DEVELOPMENT_H__
#define __DEVELOPMENT_H__

#include "environment.h"
#include "phenotype_program.h"

namespace epigenetic_gol_kernel {

// Calculate one cell value in an organism phenotype. Calling this function will
// modify the value of Cell, which occupies position (row, col) within the GOL
// board. The new value for Cell is computed by interpreting program and running
// it against the data found in genotype.
__device__ void make_phenotype(
        const PhenotypeProgram& program, const Genotype& genotype,
        const int& row, const int& col, Cell& cell);

} // namespace epigenetic_gol_kernel

#endif
