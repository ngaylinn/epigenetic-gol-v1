#include "development.cuh"

#include "cuda_utils.cuh"

namespace epigenetic_gol_kernel {
namespace {

// Special value for coordinate transforms, indicating that a Cell is no longer
// part of the logical space of the board and should not receive any value.
const int OUT_OF_BOUNDS = -1;

__device__ Scalar get_scalar(
        const Argument& arg, const Genotype& genotype) {
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            return arg.scalar_bias;
        default:
            return genotype.scalar_genes[arg.gene_index];
    }
}

__device__ const Stamp& get_stamp(
        const Argument& arg, const Genotype& genotype) {
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            return arg.stamp_bias;
        default:
            return genotype.stamp_genes[arg.gene_index];
    }
}

__device__ void apply_array_1d(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;

    int rep_number = min(row / row_offset, col / col_offset);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
}

__device__ void apply_array_2d(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;

    row %= row_offset;
    col %= col_offset;
}

__device__ void apply_copy(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;

    int rep_number = min(min(row / row_offset, col / col_offset), 1);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
}

__device__ void apply_crop(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;
    row = (row < row_offset) ? row : OUT_OF_BOUNDS;
    col = (col < col_offset) ? col : OUT_OF_BOUNDS;
}

__device__ void apply_draw(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell& cell) {
    const Stamp& stamp = get_stamp(op.args[0], genotype);
    const bool in_bounds = (row >= 0 && row < STAMP_SIZE &&
                            col >= 0 && col < STAMP_SIZE);
    cell = in_bounds ? stamp[row][col] : Cell::DEAD;
}

__device__ void apply_flip(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int axes = get_scalar(op.args[0], genotype);
    row = (axes & 0b01) ? WORLD_SIZE - 1 - row : row;
    col = (axes & 0b10) ? WORLD_SIZE - 1 - col : col;
}

__device__ void apply_mask(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    const Stamp& stamp = get_stamp(op.args[0], genotype);
    constexpr int scale = WORLD_SIZE / STAMP_SIZE;

    const bool alive = stamp[row / scale][col / scale] == Cell::ALIVE;
    row = alive ? row : OUT_OF_BOUNDS;
    col = alive ? col : OUT_OF_BOUNDS;
}

__device__ void apply_mirror(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int axes = get_scalar(op.args[0], genotype);
    row = (axes & 0b01) && row >= WORLD_SIZE / 2 ? WORLD_SIZE - 1 - row : row;
    col = (axes & 0b10) && col >= WORLD_SIZE / 2 ? WORLD_SIZE - 1 - col : col;
}

__device__ void apply_quarter(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int axes = get_scalar(op.args[0], genotype);
    constexpr int half = WORLD_SIZE / 2;
    unsigned char quadrant_bitmask = 1 << ((row >= half) | (col >= half) << 1);
    row = axes & quadrant_bitmask ? row : OUT_OF_BOUNDS;
    col = axes & quadrant_bitmask ? col : OUT_OF_BOUNDS;
}

__device__ void apply_rotate(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int rotation = get_scalar(op.args[0], genotype) % 4;
    if (rotation == 0) {
        return;
    } else if (rotation == 1) {
        int old_row = row;
        row = WORLD_SIZE - 1 - col;
        col = old_row;
    } else if (rotation == 2) {
        row = WORLD_SIZE - 1 - row;
        col = WORLD_SIZE - 1 - col;
    } else if (rotation == 3) {
        int old_row = row;
        row = col;
        col = WORLD_SIZE - 1 - old_row;
    }
}

__device__ void apply_scale(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    constexpr int max_scale = WORLD_SIZE / STAMP_SIZE;
    int row_scale = (get_scalar(op.args[0], genotype) - 1) % max_scale + 1;
    int col_scale = (get_scalar(op.args[1], genotype) - 1) % max_scale + 1;
    row = row / row_scale;
    col = col / col_scale;
}

__device__ void apply_test(
        const Genotype&, const Operation&,
        int& row, int& col, Cell& cell) {
    constexpr int half = WORLD_SIZE / 2;
    float gradient1 = row < half ? abs(float(col % 16) - 8) / 8.0 : 1.0;
    float gradient2 = float(WORLD_SIZE - col) / WORLD_SIZE;
    float gradient3 = abs(float(row) - half) / half;
    cell = Cell(255.0 * gradient1 * gradient2 * gradient3);
}

__device__ void apply_tile(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int offset = get_scalar(op.args[0], genotype) % STAMP_SIZE;
    bool flip_every_other = get_scalar(op.args[1], genotype) & 0b1;

    bool every_other_row = (row / STAMP_SIZE) & 0b1;
    offset = every_other_row ? offset : 0;
    bool every_other_col = ((col + offset) / STAMP_SIZE) & 0b1;
    bool flip = flip_every_other && (every_other_row ^ every_other_col);

    row = flip ? STAMP_SIZE - 1 - row % STAMP_SIZE : row % STAMP_SIZE;
    col = flip ? (STAMP_SIZE - 1 - col + offset) % STAMP_SIZE
               : (col + offset) % STAMP_SIZE;
}

__device__ void apply_translate(
        const Genotype& genotype, const Operation& op,
        int& row, int& col, Cell&) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;
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
        case OperationType::CROP:
            return apply_crop;
        case OperationType::DRAW:
            return apply_draw;
        case OperationType::FLIP:
            return apply_flip;
        case OperationType::MASK:
            return apply_mask;
        case OperationType::MIRROR:
            return apply_mirror;
        case OperationType::QUARTER:
            return apply_quarter;
        case OperationType::ROTATE:
            return apply_rotate;
        case OperationType::SCALE:
            return apply_scale;
        case OperationType::TEST:
            return apply_test;
        case OperationType::TILE:
            return apply_tile;
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
        const PhenotypeProgram& program, const Genotype& genotype,
        int row, int col, Cell& cell) {
    unsigned int op_index = START_INDEX;
    do {
        const Operation& op = program.ops[op_index];
        if (row != OUT_OF_BOUNDS && col != OUT_OF_BOUNDS) {
            lookup_apply(op.type)(genotype, op, row, col, cell);
        } else {
            cell = Cell::DEAD;
        }
        op_index = op.next_op_index;
    } while (op_index != STOP_INDEX);
}

} // namespace epigenetic_gol_kernel
