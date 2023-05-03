import numpy as np

import kernel


# TODO: Make private?
def get_default():
    program = PhenotypeProgram()
    program.add_operation(kernel.OperationType.TILE)
    program.add_operation(kernel.OperationType.DRAW)
    return program.data


# TODO: Find a better name?
def get_defaults(count):
    return np.full((count,), get_default(), dtype=kernel.PhenotypeProgram)


# TODO: Consider moving this to C++ to make a better API?
class PhenotypeProgram:
    def __init__(self):
        self.data = np.zeros((), dtype=kernel.PhenotypeProgram)
        self.num_ops = 0

    def add_operation(self, op_type, op_args=None):
        # Normalize the op_args argument. The caller can pass a single value, a
        # tuple, list, or nothing. The function body works on a list / tuple.
        if op_args is None:
            op_args = ()
        if not isinstance(op_args, tuple | list):
            op_args = (op_args,)

        # If there were some operations added already, make sure the last one
        # is updated so it is no longer last, but points to the new operation.
        if self.num_ops > 0:
            self.data['ops'][self.num_ops - 1]['next_op_index'] = self.num_ops

        # Populate the next available operation space in the data structure to
        # match whatever the caller specified.
        operation = self.data['ops'][self.num_ops]
        operation['type'] = op_type
        for arg_index, op_arg in enumerate(op_args):
            operation['args'][arg_index]['bias_mode'] = (
                kernel.BiasMode.FIXED_VALUE)
            if isinstance(op_arg, int):
                gene_bias = 'scalar_bias'
            else:  # assume arg is a Stamp if not an int.
                gene_bias = 'stamp_bias'
            operation['args'][arg_index][gene_bias] = op_arg

        # The new item is now the end of the program / list of operations.
        operation['next_op_index'] = kernel.STOP_INDEX
        self.num_ops += 1

    # TODO: Add utilities for breeding species.
