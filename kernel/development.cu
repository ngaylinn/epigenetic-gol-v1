#include "development.cuh"

#include "cuda_utils.cuh"
#include "environment.h"
#include "phenotype_program.h"

namespace epigenetic_gol_kernel {
namespace {

// ---------------------------------------------------------------------------
// Utilities for transform functions
// ---------------------------------------------------------------------------

// Special value for coordinate transforms, indicating that a Cell is no longer
// part of the logical space of the board and should not receive any value.
const int OUT_OF_BOUNDS = -1;

__device__ Scalar get_scalar(
        const ScalarArgument& arg, const Genotype& genotype) {
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            return arg.bias;
        default:
            return genotype.scalar_genes[arg.gene_index % NUM_GENES];
    }
}

__device__ const Stamp& get_stamp(
        const StampArgument& arg, const Genotype& genotype) {
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            return arg.bias;
        default:
            return genotype.stamp_genes[arg.gene_index % NUM_GENES];
    }
}

// ---------------------------------------------------------------------------
// Transform function definitions
// ---------------------------------------------------------------------------

__device__ void apply_align(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int v_align = get_scalar(op.args[0], genotype) % 3;
    int h_align = get_scalar(op.args[1], genotype) % 3;
    constexpr int min_edge = 0;
    constexpr int center = (WORLD_SIZE - STAMP_SIZE) / 2;
    constexpr int max_edge = WORLD_SIZE - STAMP_SIZE;

    row = (v_align == 0 ? min_edge : (v_align == 1 ? center : max_edge));
    col = (h_align == 0 ? min_edge : (h_align == 1 ? center : max_edge));
}

__device__ void apply_array_1d(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;

    int rep_number = min(row / row_offset, col / col_offset);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
}

__device__ void apply_array_2d(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;

    row %= row_offset;
    col %= col_offset;
}

__device__ void apply_copy(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int row_offset = get_scalar(op.args[0], genotype) % WORLD_SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % WORLD_SIZE;

    int rep_number = min(min(row / row_offset, col / col_offset), 1);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
}

template<int SIZE>
__device__ void apply_crop(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int row_offset = get_scalar(op.args[0], genotype) % SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % SIZE;
    row = (row < row_offset) ? row : OUT_OF_BOUNDS;
    col = (col < col_offset) ? col : OUT_OF_BOUNDS;
}

template<int SIZE>
__device__ void apply_flip(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int axes = get_scalar(op.args[0], genotype);
    row = (axes & 0b01) ? SIZE - 1 - row : row;
    col = (axes & 0b10) ? SIZE - 1 - col : col;
}

template<int SIZE>
__device__ void apply_mirror(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int axes = get_scalar(op.args[0], genotype);
    row = (axes & 0b01) && row >= SIZE / 2 ? SIZE - 1 - row : row;
    col = (axes & 0b10) && col >= SIZE / 2 ? SIZE - 1 - col : col;
}

template<int SIZE>
__device__ void apply_quarter(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int axes = get_scalar(op.args[0], genotype);
    constexpr int half = SIZE / 2;
    unsigned char quadrant_bitmask = 1 << ((row >= half) | (col >= half) << 1);
    row = axes & quadrant_bitmask ? row : OUT_OF_BOUNDS;
    col = axes & quadrant_bitmask ? col : OUT_OF_BOUNDS;
}

template<int SIZE>
__device__ void apply_rotate(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int rotation = get_scalar(op.args[0], genotype) % 4;
    if (rotation == 0) {
        return;
    } else if (rotation == 1) {
        int old_row = row;
        row = SIZE - 1 - col;
        col = old_row;
    } else if (rotation == 2) {
        row = SIZE - 1 - row;
        col = SIZE - 1 - col;
    } else if (rotation == 3) {
        int old_row = row;
        row = col;
        col = SIZE - 1 - old_row;
    }
}

__device__ void apply_scale(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    constexpr int max_scale = STAMP_SIZE;
    int row_scale = (get_scalar(op.args[0], genotype) - 1) % max_scale + 1;
    int col_scale = (get_scalar(op.args[1], genotype) - 1) % max_scale + 1;
    row = row / row_scale;
    col = col / col_scale;
}

__device__ void apply_tile(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
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

template<int SIZE>
__device__ void apply_translate(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    int row_offset = get_scalar(op.args[0], genotype) % SIZE;
    int col_offset = get_scalar(op.args[1], genotype) % SIZE;
    row = (row - row_offset) % SIZE;
    col = (col - col_offset) % SIZE;
}

template<int SIZE>
__device__ void apply_transform(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col);

// ---------------------------------------------------------------------------
// Interpret a PhenotypeProgram + Genotype to make a Phenotype
// ---------------------------------------------------------------------------

// TODO: Switch-based dispatching is kinda slow. It would be better to use
// vtable or tag-based based dispatching, but those don't apply when the
// types aren't known in advance. It would likely require a "compilation"
// pass over the program, to resolve the types once up front and embodying the
// results in a data structure or dynamically generated PTX file. It might be
// more elegant to work with the Python data structure directly rather than
// using these structs as an intermediate structure, but that may be awkward if
// using vtable lookup, since the compiled objects would have to be built on
// the GPU for that to work, which means passing Python objects to the GPU.
template<>
__device__ void apply_transform<WORLD_SIZE>(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    // Apply global transformations. Note that some transformations can be used
    // either globally or locally. They get configured by a template argument.
    switch (op.type) {
        case TransformMode::ALIGN:
            return apply_align(genotype, op, row, col);
        case TransformMode::ARRAY_1D:
            return apply_array_1d(genotype, op, row, col);
        case TransformMode::ARRAY_2D:
            return apply_array_2d(genotype, op, row, col);
        case TransformMode::COPY:
            return apply_copy(genotype, op, row, col);
        case TransformMode::CROP:
            return apply_crop<WORLD_SIZE>(genotype, op, row, col);
        case TransformMode::FLIP:
            return apply_flip<WORLD_SIZE>(genotype, op, row, col);
        case TransformMode::MIRROR:
            return apply_mirror<WORLD_SIZE>(genotype, op, row, col);
        case TransformMode::QUARTER:
            return apply_quarter<WORLD_SIZE>(genotype, op, row, col);
        case TransformMode::ROTATE:
            return apply_rotate<WORLD_SIZE>(genotype, op, row, col);
        case TransformMode::SCALE:
            return apply_scale(genotype, op, row, col);
        case TransformMode::TILE:
            return apply_tile(genotype, op, row, col);
        case TransformMode::TRANSLATE:
            return apply_translate<WORLD_SIZE>(genotype, op, row, col);
        default:
            return;
    }
}

template<>
__device__ void apply_transform<STAMP_SIZE>(
        const Genotype& genotype, const TransformOperation& op,
        int& row, int& col) {
    // Only some of the transform operations actually make sense to apply to
    // the stamp coordinate space. Any others will be ignored.
    switch (op.type) {
        case TransformMode::CROP:
            return apply_crop<STAMP_SIZE>(genotype, op, row, col);
        case TransformMode::FLIP:
            return apply_flip<STAMP_SIZE>(genotype, op, row, col);
        case TransformMode::MIRROR:
            return apply_mirror<STAMP_SIZE>(genotype, op, row, col);
        case TransformMode::QUARTER:
            return apply_quarter<STAMP_SIZE>(genotype, op, row, col);
        case TransformMode::ROTATE:
            return apply_rotate<STAMP_SIZE>(genotype, op, row, col);
        case TransformMode::SCALE:
            return apply_scale(genotype, op, row, col);
        case TransformMode::TRANSLATE:
            return apply_translate<STAMP_SIZE>(genotype, op, row, col);
        default:
            return;
    }
}

template<int SIZE>
__device__ void apply_transform_list(
        const Genotype& genotype, const TransformOperation* transforms,
        int& row, int& col) {
    // Go through the array of transforms, applying them to row and col to
    // remap the coordinate space we're drawing on.
    for (int i = 0; i < MAX_OPERATIONS; i++) {
        // A NONE transform has no effect and indicates the end of this part of
        // the program. Stop processing the transform list.
        if (transforms[i].type == TransformMode::NONE) {
            break;
        }
        // If this position has been marked out of bounds by a previous
        // transform (like a CROP operation), then no further transforms get to
        // apply here.
        if (row == OUT_OF_BOUNDS || col == OUT_OF_BOUNDS) {
            break;
        }
        // Otherwise, keep transforming row and col.
        apply_transform<SIZE>(genotype, transforms[i], row, col);
    }
}

__device__ bool apply_draw(
        const Genotype& genotype, const DrawOperation& draw_op,
        int& row, int& col) {
    // Transform the global coordinate space. This makes it possible to
    // position the stamp anywhere, repeat the stamp, and warp it in various
    // ways. Originally, row and col indicate the distance from the top-left
    // corner, but once all the transforms are applied, they indicate where in
    // the stamp to draw from.
    apply_transform_list<WORLD_SIZE>(
            genotype, draw_op.global_transforms, row, col);

    // Make sure not to read data from beyond the extents of the stamp found in
    // the genotype. Anything out of bounds for the stamp will be empty space.
    // This is computed BEFORE applying the stamp transforms so they won't
    // distort the result by modifying row and col further.
    bool in_bounds = (row >= 0 && row < STAMP_SIZE &&
                      col >= 0 && col < STAMP_SIZE);

    // Transform the stamp coordinate space. This allows for neutral mutations
    // to the data being drawn.
    apply_transform_list<STAMP_SIZE>(
            genotype, draw_op.stamp_transforms, row, col);

    // Update the in_bounds calculation after the additional transforms.
    in_bounds &= (row >= 0 && row < STAMP_SIZE &&
                  col >= 0 && col < STAMP_SIZE);

    // Actually fetch the relevant data from the genotype and determine whether
    // this cell should be set alive or not.
    const Stamp& stamp = get_stamp(draw_op.stamp, genotype);
    return in_bounds && stamp[row][col] == Cell::ALIVE;
}

} // namespace

__device__ void make_phenotype(
        const PhenotypeProgram& program, const Genotype& genotype,
        const int& row, const int& col, Cell& cell) {
    // The first draw operation always applies directly to the phenotype. After
    // that, additional draw operations layer over what came before using
    // whatever composition is specified.
    bool alive = false;
    for (int i = 0; i < MAX_OPERATIONS; i++) {
        if (program.draw_ops[i].compose_mode == ComposeMode::NONE) {
            break;
        }
        // Make a copy of row / col for this draw operation. This way, each
        // draw operation can transform the coordinate space independently.
        int r = row;
        int c = col;
        // Figure out whether this draw operation wants to set the cell to dead
        // or alive, then merge that with the previous values.
        const bool new_value = apply_draw(genotype, program.draw_ops[i], r, c);
        switch (program.draw_ops[i].compose_mode) {
            case ComposeMode::OR:
                alive |= new_value;
                break;
            case ComposeMode::XOR:
                alive ^= new_value;
                break;
            case ComposeMode::AND:
                alive &= new_value;
                break;
            default:
                break;
        };
    }
    // Actually modify the cell value based on composing all the draw
    // operations. Note, the code above uses bool values only to translate into
    // Cell values here because treating ALIVE and DEAD as bools has
    // counter-intuitive behavior. The value of ALIVE is 0x00 so that a live
    // cell appears black in the output images.
    cell = alive ? Cell::ALIVE : Cell::DEAD;
}

} // namespace epigenetic_gol_kernel
