#include "development.cuh"

#include "interpreter.cuh"

#include <climits>

namespace epigenetic_gol_kernel {
namespace {

// Argument values are just raw unsigned ints, but to perform operations we need
// values within well-specified ranges. Note, this scales values rather than
// using modulus. This is more expensive, but maintains a linear relationship
// between input and output, which should be useful for evolving argument bias.
__device__ int scale(const unsigned int value, int min, int max) {
    const unsigned int scale_factor = UINT_MAX / (max - min);
    return value / scale_factor + min;
}

__device__ int as_coord(const unsigned int value) {
    return scale(value, 0, WORLD_SIZE);
}

template<typename T>
__device__ T as_enum(const unsigned int value) {
    return (T) scale(value, 0, (int) T::SIZE);
}

__device__ void apply_array_1d(
        const BoundArguments& args, int& row, int& col, Cell&) {
    const int row_offset = as_coord(args.scalar_value(0));
    const int col_offset = as_coord(args.scalar_value(1));

    int rep_number = min(row / row_offset, col / col_offset);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
    return;
}

__device__ void apply_array_2d(
        const BoundArguments& args, int& row, int& col, Cell&) {
    const int row_offset = as_coord(args.scalar_value(0));
    const int col_offset = as_coord(args.scalar_value(1));

    row %= row_offset;
    col %= col_offset;
    return;
}

__device__ void apply_copy(
        const BoundArguments& args, int& row, int& col, Cell&) {
    const int row_offset = as_coord(args.scalar_value(0));
    const int col_offset = as_coord(args.scalar_value(1));

    int rep_number = min(min(row / row_offset, col / col_offset), 1);
    row -= rep_number * row_offset;
    col -= rep_number * col_offset;
    return;
}

__device__ void apply_draw(
        const BoundArguments& args, int& row, int& col, Cell& cell) {
    const Stamp& stamp = args.stamp_value(0);
    const bool in_bounds = (row >= 0 && row < STAMP_SIZE &&
                            col >= 0 && col < STAMP_SIZE);
    cell = in_bounds ? stamp[row][col] : Cell::DEAD;
}

__device__ void apply_translate(
        const BoundArguments& args, int& row, int& col, Cell&) {
    int row_offset = as_coord(args.scalar_value(0));
    int col_offset = as_coord(args.scalar_value(1));
    row = (row - row_offset) % WORLD_SIZE;
    col = (col - col_offset) % WORLD_SIZE;
}

__device__ void apply_none(const BoundArguments&, int&, int&, Cell&) {}

// The type specification for the per-operation apply functions.
typedef void (*apply_func)(
        const BoundArguments& args, int& row, int& col, Cell& cell);

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

__device__ void apply_operation(
        OperationType type, const BoundArguments& args,
        int& row, int& col, Cell& cell) {
    lookup_apply(type)(args, row, col, cell);
}

} // namespace epigenetic_gol_kernel
