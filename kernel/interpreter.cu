#include "interpreter.cuh"

#include <climits>

namespace epigenetic_gol_kernel {

__device__ void Argument::init(int index) {
    gene_index = index;
    bias_mode = BiasMode::NONE;
}

__device__ void Argument::init(
        int index, unsigned int scalar_bias, BiasMode mode) {
    gene_index = index;
    gene_bias.scalar = scalar_bias;
    bias_mode = mode;
}

__device__ void Argument::init(
        int index, const Stamp& stamp_bias, BiasMode mode) {
    gene_index = index;
    memcpy(&gene_bias.stamp, stamp_bias, sizeof(Stamp));
    bias_mode = mode;
}

__device__ unsigned int BoundArguments::scalar_value(int index) const {
    const Argument& arg = args[index];
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            return arg.gene_bias.scalar;
        default:
            return genotype.get_scalar(arg.gene_index);
    }
}

__device__ const Stamp& BoundArguments::stamp_value(int index) const {
    const Argument& arg = args[index];
    switch (arg.bias_mode) {
        case BiasMode::FIXED_VALUE:
            printf("Using unset stamp!\n");
            return arg.gene_bias.stamp;
        default:
            return genotype.get_stamp(arg.gene_index);
    }
}

__device__ Interpreter::Interpreter(const Interpreter& other) {
    memcpy((void*) ops, (void*) other.ops, sizeof(Operation) * MAX_OPERATIONS);
}

__device__ void Interpreter::run(
        const Genotype& genotype, int row, int col, Cell& cell) const {
    int op_index = START_INDEX;
    while (op_index != STOP_INDEX) {
        const Operation& op = ops[op_index];
        BoundArguments bound_args(op.args, genotype);
        apply_operation(op.type, bound_args, row, col, cell);
        op_index = op.next_op_index;
    }
}

__device__ DefaultInterpreter::DefaultInterpreter() {
    Operation& translate = ops[0];
    translate.type = OperationType::TRANSLATE;
    translate.args[0].init(0, 0);
    translate.args[1].init(0, 0);
    translate.next_op_index = 1;

    Operation& repeat = ops[1];
    repeat.type = OperationType::ARRAY_2D;
    repeat.args[0].init(0, 8 * (UINT_MAX / WORLD_SIZE));
    repeat.args[1].init(0, 8 * (UINT_MAX / WORLD_SIZE));
    repeat.next_op_index = 2;

    Operation& draw = ops[2];
    draw.type = OperationType::DRAW;
    draw.args[0].init(0);
    draw.next_op_index = STOP_INDEX;
}

__device__ void MockInterpreter::run(
        const Genotype&,
        int row, int col,
        Cell& cell) const {
    cell = (*injected_phenotype)[row][col];
}

} // namespace epigenetic_gol_kernel
