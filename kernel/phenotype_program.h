/*
 * Struct types for representing a program for interpreting genotypes.
 *
 * The structs below are used to represent an organism's genotype and the evolved
 * program used to interpret that genotype into a phenotype. Although tree-like
 * in structure, this data is represented as arrays of structs that get allocated
 * in a contiguous block of memory. This makes transferring data between CPU and
 * GPU simpler and more efficient.
 */

#ifndef __PHENOTYPE_PROGRAM_H__
#define __PHENOTYPE_PROGRAM_H__

#include <cstdint>

#include "environment.h"

namespace epigenetic_gol_kernel {

// This project uses two types of genes, Scalar and Stamp. Scalar genes are
// single values, used to configure the behavior of a TransformOperation's
// apply function. Stamp genes are 8x8 arrays of Cell values, used to store
// data that can be copied directly into a phenotype.
typedef unsigned int Scalar;
constexpr unsigned int STAMP_SIZE = 8;
constexpr unsigned int CELLS_PER_STAMP = STAMP_SIZE * STAMP_SIZE;
typedef Cell Stamp[STAMP_SIZE][STAMP_SIZE];

// Each organism's genotype is just a collection of Stamp and Scalar genes.
constexpr unsigned int NUM_GENES = 4;
struct Genotype {
    Scalar scalar_genes[NUM_GENES];
    Stamp stamp_genes[NUM_GENES];
};

// Fixed allocation sizes for PhenotypeProgram programs.
constexpr unsigned int MAX_OPERATIONS = 4;
constexpr unsigned int MAX_ARGUMENTS = 2;

// An enum representing how to bias data read from the genotype. For now,
// this might as well be a boolean, but it could get more interesting in
// future variations.
enum class BiasMode {
    NONE,        // Use the gene value from the genotype unmodified.
    FIXED_VALUE, // Use a fixed value instead of reading from the genotype.
    SIZE
};

// All the different kinds of TransformOperations (implemented in development.cu)
// Make sure this list stays in sync with the implementations there and the
// Python bindings in python_module.cc.
enum class TransformMode {
    NONE,
    ALIGN,     // Position stamp at the center or along an edge
    ARRAY_1D,  // Repeat coordinates in a 1D line
    ARRAY_2D,  // Repeat coordinates in a 2D grid
    COPY,      // Repeat coordinates once at some offset
    CROP,      // Ignore some portion of the available space
    // TODO: delete!
    DRAW,      // Draw a Stamp onto the phenotype
    FLIP,      // Invert coordinates horizontally and / or vertically
    MIRROR,    // Mirror half the space horizontally and / or vertically
    QUARTER,   // Blank out full quarters of the available space
    ROTATE,    // Rotate coordinates by a multiple of 90°
    SCALE,     // Scale the coordinate system up by an integral factor
    // TODO: delete!
    TEST,      // An asymmetrical test pattern for visualizing transformations
    TILE,      // Tile the coordinate space with Stamps
    TRANSLATE, // Shift coordinates by some offset
    SIZE
};

struct ScalarArgument {
    // Where in the genotype to draw data from.
    uint8_t gene_index = 0;
    // Optional override to whatever data is found in the genotype.
    Scalar bias;
    BiasMode bias_mode = BiasMode::NONE;
};

// Describes coordinate-space transformations used by DrawOperations.
struct TransformOperation {
    TransformMode type = TransformMode::NONE;
    ScalarArgument args[MAX_ARGUMENTS] = {};
};

// How to combine multiple layers of draw operations.
enum class ComposeMode {
    NONE,
    OR,
    XOR,
    AND,
    SIZE
};

struct StampArgument {
    // Where in the genotype to draw data from.
    uint8_t gene_index = 0;
    // Optional override to whatever data is found in the genotype.
    Stamp bias;
    BiasMode bias_mode = BiasMode::NONE;
};

// Describes one operation to draw patterns into phenotype, using
// data drawn from a genotype.
struct DrawOperation {
    // How to combine this with other DrawOperations 
    ComposeMode compose_mode = ComposeMode::NONE;
    // What pattern to draw
    StampArgument stamp;
    // How to transform the coordinate space before drawing the pattern.
    TransformOperation global_transforms[MAX_OPERATIONS] = {};
    TransformOperation stamp_transforms[MAX_OPERATIONS] = {};
};

struct PhenotypeProgram {
    DrawOperation draw_ops[MAX_OPERATIONS] = {};
};

} // namespace epigenetic_gol_kernel

#endif
