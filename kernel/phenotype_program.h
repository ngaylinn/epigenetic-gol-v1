#ifndef __PHENOTYPE_PROGRAM_H__
#define __PHENOTYPE_PROGRAM_H__

/*
 * Struct types for representing a program for interpreting genotypes.
 *
 * This project evolves the design for a genetic algorithm to solve a given
 * fitness goal. For that to work, the program must evolve a genotype as well
 * as a program for interpreting a genotype into a phenotype. The structs below
 * describe just such a program. They get allocated as a single block of memory
 * with a fixed size. This makes it relatively easy to pass these objects from
 * Python to C++ and from CPU to GPU.
 */

namespace epigenetic_gol_kernel {

// This project uses two types of genes, Scalar and Stamp. Scalar genes are
// single values, usually used to configure the behavior of an apply function.
// Stamp genes are 8x8 arrays of Cell values, which are useful for storing data
// to be copied into the phenotype.
typedef unsigned int Scalar;
constexpr unsigned int STAMP_SIZE = 8;
constexpr unsigned int CELLS_PER_STAMP = STAMP_SIZE * STAMP_SIZE;
typedef Cell Stamp[STAMP_SIZE][STAMP_SIZE];

// Each organism's genotype is just a collection of Stamp and Scalar genes.
constexpr unsigned int NUM_SCALARS = 4;
constexpr unsigned int NUM_STAMPS = 4;
struct Genotype {
    Scalar scalar_genes[NUM_SCALARS];
    Stamp stamp_genes[NUM_STAMPS];
};

// An enum representing how to bias data read from the genotype. This may get
// more interesting in future versions.
enum class BiasMode {
    NONE,        // Use gene value from genotype unmodified.
    FIXED_VALUE, // Use a fixed value instead of reading from the genotype.
    SIZE
};

// Bind a value from the genotype to an argument for an Operation function.
struct Argument {
    // Which gene to read from.
    int gene_index = 0;

    // Restrict genes values baked into the program to bias the results
    // read from the genotype.
    union {
        Scalar scalar;
        Stamp stamp;
    } gene_bias;
    BiasMode bias_mode = BiasMode::NONE;
};

// An exhaustive list of Operations (implemented in development.cu). When
// adding a new apply func, make sure to add a new enum value here.
enum class OperationType {
    ARRAY_1D,  // Repeat phenotype pattern in a line
    ARRAY_2D,  // Repeat phenotype pattern in a grid
    COPY,      // Copy the phenotype pattern once, with some offset
    DRAW,      // Draw a Stamp onto the phenotype
    TRANSLATE, // Shift phenotype pattern by some offset
    SIZE
};

// Fixed allocation sizes for PhenotypeProgram programs.
constexpr unsigned int MAX_OPERATIONS = 10;
constexpr unsigned int MAX_ARGUMENTS = 4;

// Program counter constants indicating where to start and stop.
constexpr unsigned int START_INDEX = 0;
constexpr unsigned int STOP_INDEX = MAX_OPERATIONS;

// A description of a single function call with arguments.
struct Operation {
    OperationType type = OperationType::SIZE;
    Argument args[MAX_ARGUMENTS] = {};
    unsigned int next_op_index = STOP_INDEX;
};

// An table of operations used to construct a phenotype.
struct PhenotypeProgram {
    Operation ops[MAX_OPERATIONS] = {};
};

} // namespace epigenetic_gol_kernel

#endif
