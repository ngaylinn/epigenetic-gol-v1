from copy import deepcopy
from dataclasses import dataclass
import itertools
import random

import numpy as np
from scipy import stats

from kernel import (
    BiasMode, ComposeMode, PhenotypeProgramDType, TransformMode,
    MAX_ARGUMENTS, MAX_OPERATIONS, NUM_GENES)

MUTATION_RATE = 0.001
CROSSOVER_RATE = 0.6

GLOBAL_TRANSFORM_OPS = [
    TransformMode.ALIGN,
    TransformMode.ARRAY_1D,
    TransformMode.ARRAY_2D,
    TransformMode.COPY,
    TransformMode.CROP,
    TransformMode.FLIP,
    TransformMode.MIRROR,
    TransformMode.QUARTER,
    TransformMode.ROTATE,
    TransformMode.SCALE,
    TransformMode.TILE,
    TransformMode.TRANSLATE,
]

STAMP_TRANSFORM_OPS = [
    TransformMode.CROP,
    TransformMode.FLIP,
    TransformMode.MIRROR,
    TransformMode.QUARTER,
    TransformMode.ROTATE,
    TransformMode.SCALE,
    TransformMode.TRANSLATE,
]


def coin_flip(probability=0.5):
    return random.random() < probability


def crossover_operation_lists(op_list_a, op_list_b):
    class Edge:
        def __init__(self):
            self.inno = None

    # If the operation lists are disjoint, consider concatenating them instead
    # of choosing one at random, which is what would happen otherwise.
    disjoint = len(
        set(op_a.inno for op_a in op_list_a) &
        set(op_b.inno for op_b in op_list_b)) == 0
    if disjoint:
        options = [op_list_a, op_list_b]
        if len(op_list_a) + len(op_list_b) < MAX_OPERATIONS:
            options.extend([op_list_a + op_list_b,
                            op_list_b + op_list_a])
        return deepcopy(random.choice(options))

    # A summary of all the operations found in op_list_a and op_list_b, keyed
    # by innovation number. This is used to group together variants of the
    # "same" operation that appear in both parent lists. Operations that share
    # the same innovation number were created in the same mutation event and in
    # some sense serve the same "purpose" in the PhenotypeProgram even though
    # they may be configured differently.
    ops = {}
    # A mapping from innovation number to innovation number representing which
    # operations follow each other in the input lists. The result of crossover
    # must preserve the original order of operations from one or both inputs.
    # Dict entries with key None must go at the start of the list, while
    # entries with value None must go at the end.
    links = {}

    # Go through both input lists to populate ops and links. To account for the
    # beginning and end of the list in the links dictionary, add "null
    # terminators" to both ends of the input lists.
    op_list_a = [Edge()] + op_list_a + [Edge()]
    for (from_op, to_op) in itertools.pairwise(op_list_a):
        ops.setdefault(from_op.inno, set()).update({from_op})
        links.setdefault(from_op.inno, set()).update({to_op.inno})
    op_list_b = [Edge()] + op_list_b + [Edge()]
    for (from_op, to_op) in itertools.pairwise(op_list_b):
        ops.setdefault(from_op.inno, set()).update({from_op})
        links.setdefault(from_op.inno, set()).update({to_op.inno})

    # Build up a new operation list beginning at start and repeatedly choosing
    # a valid next operation from either parent op_list.
    result = []
    next_op_options = list(links[None])
    # Repeat as long as there's a valid next operation to select.
    while next_op_options:
        # Randomly select an operation that could follow the previous one.
        next_op_inno = random.choice(next_op_options)
        next_op_options.remove(next_op_inno)

        # If the selected next operation is the null terminator, then the end
        # was reached. Stop and return what was built so far.
        if next_op_inno is None:
            break

        # If both parents have a copy of the same operation, perform crossover
        # on that operation node and use the result.
        op_variants = list(ops[next_op_inno])
        if len(op_variants) == 2:
            next_op = op_variants[0].crossover(op_variants[1])
        # If this operation came from just one parent, and there's another
        # operation that could take this position, then maybe pass over this
        # one (with 50% probability) and take the next one instead.
        elif next_op_options and coin_flip():
            continue
        # Otherwise, this is the only valid operation at this position.
        else:
            next_op = op_variants[0]

        # Add next_op to the sequence, consider what operations are allowed to
        # follow that one, and iterate.
        result.append(deepcopy(next_op))
        next_op_options = list(links[next_op_inno])
    return result


@dataclass
class Constraints:
    allow_bias: bool = False
    allow_composition: bool = False
    allow_stamp_transforms: bool = False


class Argument:
    def __init__(self):
        # By default all arguments reference the same gene. This means all
        # arguments of all operations will have the same value until mutations
        # cause them to diverge. This allows coordinated behavior and graduated
        # complexity in the population.
        self.gene_index = 0
        # By default, operations take their arguments from the genotype without
        # any evolved preference for what the value should be.
        self.bias_mode = BiasMode.NONE
        self.bias = None

    def mutate(self, bias, constraints, mutation_rate=MUTATION_RATE):
        if coin_flip(mutation_rate):
            self.gene_index = random.randrange(NUM_GENES)
        if (constraints.allow_bias and bias is not None and
                coin_flip(mutation_rate)):
            # Randomly select a different bias mode than the current one.
            max_bias = BiasMode.SIZE.value
            bias_options = list(set(range(max_bias)) - {self.bias_mode.value})
            self.bias_mode = BiasMode(random.choice(bias_options))
            if self.bias_mode != BiasMode.NONE:
                self.bias = bias

    def serialize(self, output_array):
        output_array['gene_index'] = self.gene_index
        output_array['bias_mode'] = self.bias_mode
        if self.bias is not None:
            output_array['bias'] = self.bias


class TransformOperation:
    def __init__(self, inno, transform_type=None, transform_args=None,
                 type_options=None):
        self.inno = inno
        if all((transform_type, transform_args)):
            self.type = transform_type
            self.args = transform_args
        else:
            # If type_optins was passed in, randomly choose from one of those
            # types.
            if type_options:
                self.type = random.choice(type_options)
            else:
                self.type = TransformMode.TRANSLATE
            # Args always starts with default settings, even when the type is
            # randomized. This encourages simple organisms by default, with
            # greater complexity added via mutations.
            self.args = [Argument() for _ in range(MAX_ARGUMENTS)]

    def crossover(self, other):
        assert self.inno == other.inno
        return TransformOperation(
            self.inno,
            random.choice((self.type, other.type)),
            [random.choice((arg_a, arg_b))
             for arg_a, arg_b in zip(self.args, other.args)])

    def mutate(self, genotypes, constraints, type_options,
               mutation_rate=MUTATION_RATE):
        if coin_flip(mutation_rate):
            self.type = random.choice(type_options)
        for arg in self.args:
            # Look at the relevant scalar gene for every organism of this
            # species and take the mode. That means a common value for this
            # gene will be taken from the population and used as the preferred
            # value for this argument in the next generation.
            evolved_scalars = genotypes['scalar_genes'][:, arg.gene_index]
            bias = stats.mode(evolved_scalars, keepdims=False).mode
            arg.mutate(bias, constraints, mutation_rate)

    def randomize(self):
        for arg in self.args:
            arg.gene_index = random.randrange(NUM_GENES)

    def serialize(self, output_array):
        output_array['type'] = self.type
        for index, arg in enumerate(self.args):
            arg.serialize(output_array['args'][index])

    def __str__(self):
        args = ', '.join([
            f'{arg.gene_index} ← {arg.bias}'
            if arg.bias_mode == BiasMode.FIXED_VALUE
            else f'{arg.gene_index}'
            for arg in self.args])
        return f'{self.type.name}{{{self.inno}}}({args})'


class DrawOperation:
    def __init__(self, inno, compose_mode=None, stamp=None,
                 global_transforms=None, stamp_transforms=None):
        self.inno = inno
        # Have to count None to support [] for one of the transforms args.
        all_args = [compose_mode, stamp, global_transforms, stamp_transforms]
        if all_args.count(None) == 0:
            self.compose_mode = compose_mode
            self.stamp = stamp
            self.global_transforms = global_transforms
            self.stamp_transforms = stamp_transforms
        else:
            self.compose_mode = ComposeMode.OR
            self.stamp = Argument()
            self.global_transforms = []
            self.stamp_transforms = []

    def add_global_transform(self, innovation_counter):
        transform = TransformOperation(next(innovation_counter))
        self.global_transforms.append(transform)
        return transform

    def add_stamp_transform(self, innovation_counter):
        transform = TransformOperation(next(innovation_counter))
        self.stamp_transforms.append(transform)
        return transform

    def crossover(self, other):
        assert self.inno == other.inno
        return DrawOperation(
            self.inno,
            random.choice((self.compose_mode, other.compose_mode)),
            random.choice((self.stamp, other.stamp)),
            crossover_operation_lists(
                self.global_transforms, other.global_transforms
            )[:MAX_OPERATIONS],
            crossover_operation_lists(
                self.stamp_transforms, other.stamp_transforms
            )[:MAX_OPERATIONS])

    def fork(self, innovation_counter):
        # When adding a new DrawOperation by mutation, copy an existing
        # DrawOperation and compose using an OR operation. This should be a
        # neutral mutation with no impact on the rendered phenotype, but will
        # allow divergance after future mutations.
        other = DrawOperation(
            next(innovation_counter),
            ComposeMode.OR,
            deepcopy(self.stamp),
            deepcopy(self.global_transforms),
            deepcopy(self.stamp_transforms)
        )
        # Put tne new draw operation after the original, which assures
        # composition with OR will have no effect.
        return [self, other]

    def mutate(self, innovation_counter, genotypes, constraints,
               mutation_rate=MUTATION_RATE):
        # Maybe mutate compose_mode
        if coin_flip(mutation_rate):
            # Randomly pick a compose mode (that isn't NONE)
            self.compose_mode = ComposeMode(
                random.randrange(1, ComposeMode.SIZE.value))

        # Look at the relevant stamp gene for every organism of this species
        # and take the mode and maybe use that as bias in the next generation
        # That would mean a cell will be biased towards ALIVE if most organisms
        # in the population had an ALIVE cell in that position.
        evolved_stamps = genotypes['stamp_genes'][:, self.stamp.gene_index]
        bias = stats.mode(evolved_stamps, keepdims=False).mode
        # If the genotype data produces a blank stamp (this happens in the
        # initial generation), then don't use it for bias, since it would
        # effectively disable this DrawOperation.
        if np.count_nonzero(bias) == 0:
            bias = None
        self.stamp.mutate(bias, constraints, mutation_rate)

        # Maybe add new global or stamp transform
        if (coin_flip(mutation_rate) and
                len(self.global_transforms) + 1 < MAX_OPERATIONS):
            self.global_transforms.append(
                TransformOperation(innovation_counter, GLOBAL_TRANSFORM_OPS))
        if (constraints.allow_stamp_transforms and coin_flip(mutation_rate) and
                len(self.stamp_transforms) + 1 < MAX_OPERATIONS):
            self.stamp_transforms.append(
                TransformOperation(innovation_counter, STAMP_TRANSFORM_OPS))

        # Maybe mutate one of the existing transforms
        for transform in self.global_transforms:
            transform.mutate(
                genotypes, constraints, GLOBAL_TRANSFORM_OPS, mutation_rate)
        for transform in self.stamp_transforms:
            transform.mutate(
                genotypes, constraints, STAMP_TRANSFORM_OPS, mutation_rate)

    def randomize(self, innovation_counter, constraints):
        self.stamp.gene_index = random.randrange(NUM_GENES)
        transform_type = random.choice(GLOBAL_TRANSFORM_OPS + [None])
        if transform_type is not None:
            transform = self.add_global_transform(innovation_counter)
            transform.type = transform_type
        if constraints.allow_stamp_transforms:
            transform_type = random.choice(STAMP_TRANSFORM_OPS + [None])
            if transform_type is not None:
                transform = self.add_stamp_transform(innovation_counter)
                transform.type = transform_type
        for transform in self.global_transforms:
            transform.randomize()
        for transform in self.stamp_transforms:
            transform.randomize()


    def serialize(self, output_array):
        output_array['compose_mode'] = self.compose_mode
        self.stamp.serialize(output_array['stamp'])
        for index, transform in enumerate(self.global_transforms):
            transform.serialize(output_array['global_transforms'][index])
        for index, transform in enumerate(self.stamp_transforms):
            transform.serialize(output_array['stamp_transforms'][index])

    def __str__(self):
        compose = self.compose_mode.name
        globals = (', '.join(map(str, self.global_transforms))
                   if self.global_transforms else 'NOOP')
        stamps = (', '.join(map(str, self.stamp_transforms))
                  if self.stamp_transforms else 'NOOP')
        stamp_bias = ('\n  ' + str(self.stamp.bias)
                      if self.stamp.bias_mode == BiasMode.FIXED_VALUE
                      else '')
        return (f'{compose}{{{self.inno}}}({self.stamp.gene_index})\n'
                f'  {globals}\n'
                f'   ⤷ {stamps}{stamp_bias}')


class PhenotypeProgram:
    def __init__(self, draw_ops=None):
        if draw_ops is None:
            # A PhenotypeProgram is expected to always have at least one
            # DrawOperation, and calling mutate() will fail if it does not.
            # When manually constructing a PhenotypeProgram for testing, make
            # sure to always call add_draw() after construction.
            self.draw_ops = []
        else:
            self.draw_ops = draw_ops

    def add_draw(self, innovation_counter):
        draw_op = DrawOperation(next(innovation_counter))
        self.draw_ops.append(draw_op)
        return draw_op

    def make_offspring(self, other, innovation_counter, genotypes, constraints):
        if self is not other and coin_flip(CROSSOVER_RATE):
            result = self.crossover(other)
        else:
            result = deepcopy(self)
        result.mutate(innovation_counter, genotypes, constraints)
        return result

    def crossover(self, other):
        return PhenotypeProgram(
            crossover_operation_lists(
                self.draw_ops, other.draw_ops
            )[:MAX_OPERATIONS])

    def mutate(self, innovation_counter, genotypes, constraints,
               mutation_rate=MUTATION_RATE):
        # If it's supported, and the draw list isn't already at max length,
        # maybe duplicate a DrawOperation. By using duplication instead of
        # randomization, the mutation can be neutral and more likely to persist
        # as potential variation in the population.
        if (constraints.allow_composition and coin_flip(mutation_rate) and
                len(self.draw_ops) + 1 < MAX_OPERATIONS):
            position = random.randrange(len(self.draw_ops))
            before = self.draw_ops[:position]
            middle = self.draw_ops[position].fork(innovation_counter)
            after = self.draw_ops[position + 1:]
            self.draw_ops = before + middle + after
        for draw_op in self.draw_ops:
            draw_op.mutate(
                innovation_counter, genotypes, constraints, mutation_rate)

    def randomize(self, innovation_counter, constraints):
        if constraints.allow_composition:
            compose_mode = ComposeMode(
                random.randrange(ComposeMode.SIZE.value))
            if (compose_mode != ComposeMode.NONE):
                draw_op = self.add_draw(innovation_counter)
                draw_op.compose_mode = compose_mode
        for draw_op in self.draw_ops:
            draw_op.randomize(innovation_counter, constraints)

    def serialize(self, output_array=None):
        if output_array is None:
            output_array = np.zeros((), dtype=PhenotypeProgramDType)
        for index, draw_op in enumerate(self.draw_ops):
            draw_op.serialize(output_array['draw_ops'][index])
            # Avoid no-op PhenotypePrograms by ensuring that at least one draw
            # operation always applies.
            if (index == 0):
                draw_op.compose_mode = ComposeMode.OR
        return output_array

    def __str__(self):
        draws = '\n'.join(map(str, self.draw_ops))
        return f'<\n{draws}\n>'


