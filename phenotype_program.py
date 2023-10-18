"""Classes for manipulating C++ PhenotypeProgram structs from Python.

This project uses a PhenotypeProgram struct to describe a species-specific way
of turning a genotype into a phenotype. These objects have two representations.
The kernel module defines C++ structs to store PhenotypeProgram data, and uses
those structs to generate phenotypes in development.cu. This side of the code
is designed to be relatively efficient, since it must be run once for every
simulated organism lifetime.

This module provides Python objects that represent the same data. The evolution
module designes PhenotypePrograms in an evolutionary process, using the
representations below for randomization, mutatation, and cross-over. The Python
code is optimized for readability rather than performance, since these
operations are complicated and happen only once for each species generation
(which involves thousands of organism lifetimes run with kernel.Simulator).

PhenotypePrograms in Python support "pretty printing" via the str() function.
The output looks something like this:
<
COMPOSE_MODE{innovation_number}(gene_index)
  GLOBAL_TRANSFORM_MODE{innovation_number}(arg1_gene_index ← scalar_bias, arg2_gene_index), ...
   ⤷ STAMP_TRANSFORM_MODE{innovation_number}(arg1_gene_index, arg_gen_index ← scalar_bias), ...
  [[ stamp_bias ]]
COMPOSE_MODE{innovation_number}(gene_index)
...
>
"""

from copy import deepcopy
from dataclasses import dataclass
import itertools
import random

import numpy as np
from scipy import stats

from kernel import (
    BiasMode, ComposeMode, PhenotypeProgramDType, TransformMode,
    MAX_ARGUMENTS, MAX_OPERATIONS, NUM_GENES)


# Hyperparameters for species evolution.
MUTATION_RATE = 0.001
CROSSOVER_RATE = 0.6


# Transform operations that can be applied to the phenotype globally. This list
# must be kept consistent with apply_transform in kernel/development.cu.
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

# Transform operations that can be applied to a stamp gene. This list must be
# kept consistent with apply_transform in kernel/development.cu.
STAMP_TRANSFORM_OPS = [
    TransformMode.CROP,
    TransformMode.FLIP,
    TransformMode.MIRROR,
    TransformMode.QUARTER,
    TransformMode.ROTATE,
    TransformMode.SCALE,
    TransformMode.TRANSLATE,
]


def coin_flip(probability_of_true=0.5):
    """Return a random boolean value, with the given chance of being True."""
    return random.random() < probability_of_true


def crossover_operation_lists(op_list_a, op_list_b):
    """Align and crossover two operation lists using innovation numbers.

    PhenotypePrograms are tree shaped and relatively complicated, which makes
    recombining them from two parents tricky. To understand this better, check
    out the README and the code in kernel/phentoype_program.h and
    kernel/development.cu. In short, each PhenotypeProgram is tree-shaped, and
    this is represented using nested lists of operations. Each program consists
    of a list of DrawOperation, which has two lists of TransformOperations,
    one to apply to the phenotype globally and another to apply to the Stamp
    gene referred to by the DrawOperation.

    This function can work on any of these operation lists, and treats them the
    same. The result is a new operation list which contains only operations
    from the two op_list arguments, in an order that is consistent with both
    lists. Two operations are considered "the same" if they have the same
    innovation number, meaning they trace back to the same mutation.
    Importantly, these Operation objects may not actually be identical, since
    they may also contain internal mutations. An operation that only appears in
    one of the input lists will be included with ~50% probability. If an
    operation appears in both lists, it will be included and the Operation's
    crossover method will be used to combine values from both op_lists.
    """
    # Represents either the beginning or the end of an operation list.
    class Edge:
        def __init__(self):
            self.inno = None

    # Edge case: If the operation lists are completely disjoint, consider
    # concatenating them instead of just randomly picking one list to keep,
    # which is what would happen otherwise.
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
    # some sense serve the same "purpose" in the PhenotypeProgram, even though
    # they may be configured differently.
    ops = {}
    # A mapping from innovation number to innovation number, representing which
    # operations follow each other in the input lists. The result of crossover
    # must preserve the original order of operations from one or both inputs.
    # Dict entries with key None must go at the start of the list, while
    # entries with value None must go at the end.
    links = {}

    # Go through both input lists to populate ops and links. To account for the
    # beginning and end of the list in the links dictionary, use the Edge class
    # to add null terminators to both ends of the input lists.
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
        # on that operation and use the result.
        op_variants = list(ops[next_op_inno])
        if len(op_variants) == 2:
            next_op = op_variants[0].crossover(op_variants[1])
        # If this operation came from just one parent, and there's another
        # operation that could take this position, then maybe pass over this
        # one (with 50% probability) and take the next one instead.
        elif next_op_options and coin_flip():
            continue
        # Otherwise, this is the only valid operation at this position, so use
        # it even though it comes from just one parent.
        else:
            next_op = op_variants[0]

        # Add next_op to the sequence, consider what operations are allowed to
        # follow that one, and iterate.
        result.append(deepcopy(next_op))
        next_op_options = list(links[next_op_inno])
    return result


@dataclass
class Constraints:
    """Configuration options for generating PhenotypePrograms."""
    allow_bias: bool = False
    """Whether genes can have biased values that don't evolve randomly."""
    allow_composition: bool = False
    """Whether programs can have more than one DrawOperation."""
    allow_stamp_transforms: bool = False
    """Whether Stamp genes can have TransformOperations."""


class Argument:
    """Configuration options for an Operation (Draw or Transform).

    Each Operation can be thought of as a function call, with argument values
    drawn from an organism's Genotype. The Argument class represents that
    binding between an operation argument and Genotype data.

    Attributes
    ----------
    gene_index : int
        Which gene in the genotype to draw data from for this argument.
    bias_mode : kernel.BiasMode
        How the argument value should be biased.
    bias : Stamp or Scalar
        A value for this argument that may be used instead of drawing data from
        the Genotype, depending on the bias_mode setting. This value is either
        a Stamp or a Scalar depending on whether the Argument applies to a
        DrawOperation or a TransformOperation.
    """
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
        """Maybe randomly change Argument metadata."""
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
        """Populate output_array with the data from this object."""
        output_array['gene_index'] = self.gene_index
        output_array['bias_mode'] = self.bias_mode
        if self.bias is not None:
            output_array['bias'] = self.bias


class TransformOperation:
    """Configuration options for a TransformOperation.

    Each TransformOperation can be thought of as a function that warps the
    coordinate system of the phenotype globally or for a Stamp. There are
    several different possible transforms, identified by type, each of which
    takes one or two Arguments.

    Attributes
    ----------
    inno : int
        The innovation number for this TransformOperation, indicating the
        mutation where it was first added to the PhenotypeProgram.
    type : kernel.TransformMode
        Which transformation to apply.
    args : list of Argument
        The arguments bindings passed to the transformation apply function.
    """
    def __init__(self, inno, transform_type=None, transform_args=None,
                 type_options=None):
        self.inno = inno
        # Construct from existing type and args if given.
        if all((transform_type, transform_args)):
            self.type = transform_type
            self.args = transform_args
        # Otherwise, set type and args randomly.
        else:
            # If type_options was specified, randomly choose one of those. This
            # is used to restrict to just the TransformOperations that apply
            # globally or to Stamp genes.
            if type_options:
                self.type = random.choice(type_options)
            # Otherwise, just pick a sensible default.
            else:
                self.type = TransformMode.TRANSLATE
            # Args always starts with default settings, even when the type is
            # randomized. This encourages simple organisms by default, with
            # greater complexity added via mutations.
            self.args = [Argument() for _ in range(MAX_ARGUMENTS)]

    def crossover(self, other):
        """Create a new TransformOperation by remixing two existing ones."""
        # Double check that the alignment process worked and both
        # TransformOperations arose from the same initial mutation.
        assert self.inno == other.inno
        # Randomly choose type and arg values from either parent.
        return TransformOperation(
            self.inno,
            random.choice((self.type, other.type)),
            [random.choice((arg_a, arg_b))
             for arg_a, arg_b in zip(self.args, other.args)])

    def mutate(self, genotypes, constraints, type_options,
               mutation_rate=MUTATION_RATE):
        """Maybe randomly modify a TransformOperation.

        In order to support gene bias, the genotypes from a whole population of
        organisms evolved with this PhenotypeProgram are passed in. A bias value
        may be chosen at random from the evolved gene values.
        """
        if coin_flip(mutation_rate):
            self.type = random.choice(type_options)
        for arg in self.args:
            # Select a gene value from the prior generation used for this
            # Argument to use as bias. The mode of gene values from the last
            # generation is used, which ensures the bias value is one found
            # commonly among the best evolved organisms.
            evolved_scalars = genotypes['scalar_genes'][:, arg.gene_index]
            bias = stats.mode(evolved_scalars, keepdims=False).mode
            arg.mutate(bias, constraints, mutation_rate)

    def randomize(self):
        """Randomize this Operation for use in an initial population."""
        for arg in self.args:
            arg.gene_index = random.randrange(NUM_GENES)

    def serialize(self, output_array):
        """Populate output_array with the data from this object."""
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
    """Configuration options for a DrawOperation.

    Each DrawOperation takes Stamp data from the Genotype and paints it into
    the phenotype. TransformOperations can be used to constrain the Stamp data
    or where it is placed in the phenotype, leading to features like repetition
    and symmetry. Multiple DrawOperations can be combined, by using boolean
    operators to merge multiple "layers" into a single "image".

    Attributes
    ----------
    inno : int
        The innovation number for this DrawOperation, indicating the mutation
        where it was first added to the PhenotypeProgram.
    compose_mode : kernel.ComposeMode
        How to combine this DrawOperation with the one that came before it (or
        the blank phenotype, in the case of the first DrawOperation).
    stamp : Argument
        The Stamp gene to draw data from.
    global_transforms : list of TransformOperation
        A list of TransformOperations to apply to the phenotype globally.
    stamp_transforms : list of TransformOperation
        A list of TransformOperations to apply to stamp before drawing it.
    """
    def __init__(self, inno, compose_mode=None, stamp=None,
                 global_transforms=None, stamp_transforms=None):
        self.inno = inno
        # Take the value of all attributes from the arguments if they were
        # passed in. Note that this code counts the number of None values
        # to distinguish between when one of the _transforms arguments is []
        # and when it is not passed in at all.
        all_args = [compose_mode, stamp, global_transforms, stamp_transforms]
        if all_args.count(None) == 0:
            self.compose_mode = compose_mode
            self.stamp = stamp
            self.global_transforms = global_transforms
            self.stamp_transforms = stamp_transforms
        # Otherwise, use sensible default values. This encourages the algorithm
        # to start with simple species, which accumulate complexity only when
        # it improves fitness.
        else:
            self.compose_mode = ComposeMode.OR
            self.stamp = Argument()
            self.global_transforms = []
            self.stamp_transforms = []

    def add_global_transform(self, innovation_counter):
        """Add a new global transform with a new innovation number."""
        transform = TransformOperation(next(innovation_counter))
        self.global_transforms.append(transform)
        return transform

    def add_stamp_transform(self, innovation_counter):
        """Add a new Stamp transform with a new innovation number."""
        transform = TransformOperation(next(innovation_counter))
        self.stamp_transforms.append(transform)
        return transform

    def crossover(self, other):
        """Create a new DrawOperation by remixing two existing ones."""
        # Double check that the alignment process worked and both
        # DrawOperations arose from the same initial mutation.
        assert self.inno == other.inno
        # Randomly choose attribute values from either parent, recursively
        # performing crossover on the transform lists.
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
        """Make a copy of this DrawOperation with a new innovation number."""
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
        """Maybe randomly modify a TransformOperation.

        In order to support gene bias, the genotypes from a whole population of
        organisms evolved with this PhenotypeProgram are passed in. A bias value
        may be chosen at random from the evolved gene values.
        """
        # Maybe randomly pick a new compose_mode (that isn't NONE)
        if coin_flip(mutation_rate):
            self.compose_mode = ComposeMode(
                random.randrange(1, ComposeMode.SIZE.value))

        # Choose a bias value that might be applied to the Stamp Argument. Look
        # at the relevant Stamp gene for every organism of this species, take
        # the mode, and maybe use that as bias in the next generation. That
        # means a cell will be biased towards ALIVE if most organisms in the
        # population had an ALIVE cell in that position.
        evolved_stamps = genotypes['stamp_genes'][:, self.stamp.gene_index]
        bias = stats.mode(evolved_stamps, keepdims=False).mode
        # If the genotype data produces a blank stamp (this always happens in
        # the initial generation), then don't use it for bias, since it would
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
        """Randomize this Operation for use in an initial population."""
        self.stamp.gene_index = random.randrange(NUM_GENES)
        transform_type = random.choice(GLOBAL_TRANSFORM_OPS + [None])
        # Maybe randomly add one transform of each type, if allowed.
        if transform_type is not None:
            transform = self.add_global_transform(innovation_counter)
            transform.type = transform_type
        if constraints.allow_stamp_transforms:
            transform_type = random.choice(STAMP_TRANSFORM_OPS + [None])
            if transform_type is not None:
                transform = self.add_stamp_transform(innovation_counter)
                transform.type = transform_type
        # Randomize any transforms that got added.
        for transform in self.global_transforms:
            transform.randomize()
        for transform in self.stamp_transforms:
            transform.randomize()


    def serialize(self, output_array):
        """Populate output_array with the data from this object."""
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
    """Configuration describing how to transform a Genotype to a phenotype.

    Attributes
    ----------
    draw_ops : list of DrawOperation
        The DrawOperations that make up this PhenotypeProgram.
    """
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
        """Extend this program with a new DrawOperation."""
        draw_op = DrawOperation(next(innovation_counter))
        self.draw_ops.append(draw_op)
        return draw_op

    def make_offspring(self, other, innovation_counter, genotypes, constraints):
        """Generate a new PhenotypeProgram from this one (and maybe another)."""
        # Maybe crossover self and other (unless they are the same, in which
        # case reproduce asexually).
        if self is not other and coin_flip(CROSSOVER_RATE):
            result = self.crossover(other)
        else:
            result = deepcopy(self)
        # Mutate the new child, then return it.
        result.mutate(innovation_counter, genotypes, constraints)
        return result

    def crossover(self, other):
        """Create a new PhenotypeProgram by remixing two existing ones."""
        return PhenotypeProgram(
            crossover_operation_lists(
                self.draw_ops, other.draw_ops
            )[:MAX_OPERATIONS])

    def mutate(self, innovation_counter, genotypes, constraints,
               mutation_rate=MUTATION_RATE):
        """Maybe randomly modify a TransformOperation."""
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
        # Mutate each of the draw operations.
        for draw_op in self.draw_ops:
            draw_op.mutate(
                innovation_counter, genotypes, constraints, mutation_rate)

    def randomize(self, innovation_counter, constraints):
        """Randomize this PhenotypeProgram for use in an initial population."""
        if constraints.allow_composition:
            compose_mode = ComposeMode(
                random.randrange(ComposeMode.SIZE.value))
            if (compose_mode != ComposeMode.NONE):
                draw_op = self.add_draw(innovation_counter)
                draw_op.compose_mode = compose_mode
        for draw_op in self.draw_ops:
            draw_op.randomize(innovation_counter, constraints)

    def serialize(self, output_array=None):
        """Populate output_array with the data from this object.

        If output_array is not specified, a new numpy array is constructed.
        """
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


