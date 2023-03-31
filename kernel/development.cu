#include "development.cuh"

#include <climits>

namespace epigenetic_gol_kernel {
namespace {

// Argument values are just raw unsigned ints, but to perform operations we need
// values within well-specified ranges. Note, this scales values rather than
// using modulus. This is more expensive, but maintains a linear relationship
// between input and output, which should be useful for evolving argument bias.
__device__ int scale(const Scalar value, int min, int max) {
    const unsigned int scale_factor = UINT_MAX / (max - min);
    return value / scale_factor + min;
}

__device__ int as_coord(const Scalar value) {
    return scale(value, 0, WORLD_SIZE);
}

__device__ Scalar get_scalar(
        const Argument& arg, const Genotype& genotype) {
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            return arg.gene_bias.scalar;
        default:
            return genotype.scalar_genes[arg.gene_index];
    }
}

__device__ const Stamp& get_stamp(
        const Argument& arg, const Genotype& genotype) {
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            return arg.gene_bias.stamp;
        default:
            return genotype.stamp_genes[arg.gene_index];
    }
}

__device__ void apply_array_1d(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = as_coord(get_scalar(op.args[0], genotype));
    int col_offset = as_coord(get_scalar(op.args[1], genotype));

    int rep_number = min(row / row_offset, col / col_offset);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
}

__device__ void apply_array_2d(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = as_coord(get_scalar(op.args[0], genotype));
    int col_offset = as_coord(get_scalar(op.args[1], genotype));

    row %= row_offset;
    col %= col_offset;
}

__device__ void apply_copy(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = as_coord(get_scalar(op.args[0], genotype));
    int col_offset = as_coord(get_scalar(op.args[1], genotype));

    int rep_number = min(min(row / row_offset, col / col_offset), 1);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
}

__device__ void apply_draw(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell& cell) {
    const Stamp& stamp = get_stamp(op.args[0], genotype);
    const bool in_bounds = (row >= 0 && row < STAMP_SIZE &&
                            col >= 0 && col < STAMP_SIZE);
    cell = in_bounds ? stamp[row][col] : Cell::DEAD;
}

__device__ void apply_translate(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = as_coord(get_scalar(op.args[0], genotype));
    int col_offset = as_coord(get_scalar(op.args[1], genotype));
    row = (row - row_offset) % WORLD_SIZE;
    col = (col - col_offset) % WORLD_SIZE;
}

__device__ void apply_none(
        const Genotype&, const Operation&, int&, int&, Cell&) {}

// The type specification for the per-operation apply functions.
typedef void (*apply_func)(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell& cell);

// Every time a new apply function is implemented, it must be added to this
// dispatcher here and to the OperationType enum in phenotype_program.h
__device__ apply_func lookup_apply(OperationType type) {
    switch (type) {
        case OperationType::ARRAY_1D:
            return apply_array_1d;
        case OperationType::ARRAY_2D:
            return apply_array_2d;
        case OperationType::COPY:
            return apply_copy;
        case OperationType::DRAW:
            return apply_draw;
        case OperationType::TRANSLATE:
            return apply_translate;
        default:
            return apply_none;
    }
}

} // namespace

// Note that the args row and col are both passed by value instead of by
// reference. This creates two new int objects for every call, which are
// initially set to reflect the position of cell within the phenotype. By
// decoupling these values from the original row and column, the apply
// functions can modify the indexing scheme of the phenotype being generated.
__device__ void make_phenotype(
        const Genotype& genotype, const PhenotypeProgram& program,
        int row, int col, Cell& cell) {
    unsigned int op_index = START_INDEX;
    do {
        const Operation& op = program.ops[op_index];
        lookup_apply(op.type)(genotype, op, row, col, cell);
        op_index = op.next_op_index;
    } while (op_index != STOP_INDEX);
}

} // namespace epigenetic_gol_kernel
