import itertools
import random
import unittest

import numpy as np

import experiments
from kernel import BiasMode, GenotypeDType
from phenotype_program import (
    crossover_operation_lists,
    Clade, Constraints, PhenotypeProgram)
from tests import test_case


def make_dummy_genotype():
    result = np.empty(1, dtype=GenotypeDType)
    result['scalar_genes'] = np.random.randint(0, 65, (4,), dtype=np.uint32)
    result['stamp_genes'] = (
        np.random.randint(0, 2, (4, 8, 8), dtype=np.uint8) * 255)
    return result


DUMMY_GENOTYPE = make_dummy_genotype()


def crossover(list_a, list_b):
    class MockOperation:
        def __init__(self, inno):
            self.inno = inno

        def crossover(self, other):
            return self

    result = crossover_operation_lists(
        [MockOperation(inno) for inno in list_a],
        [MockOperation(inno) for inno in list_b])
    return [operation.inno for operation in result]


def all_possible_crossovers(list_a, list_b):
    result = set()
    # The crossover process is stochastic, so to get a complete summary of
    # possible outputs just run the function repeatedly until all possible
    # outputs are exhausted. A small number of repetitions is enough since the
    # test cases are also small and produce few possible outputs.
    for _ in range(20):
        result.update({tuple(crossover(list_a, list_b))})
    return result


def count_diffs(array_a, array_b):
    names = array_a.dtype.names
    if names is None:
        return np.count_nonzero(array_a != array_b)
    result = 0
    for name in names:
        result += count_diffs(array_a[name], array_b[name])
    return result


def all_arguments(program):
    result = []
    for draw_op in program.draw_ops:
        result.append(draw_op.stamp)
        for transform in draw_op.global_transforms:
            result.extend(transform.args)
        for transform in draw_op.stamp_transforms:
            result.extend(transform.args)
    return result


class TestPhenotypeProgram(test_case.TestCase):
    def setUp(self):
        random.seed(42)

    def test_crossover_same_structure(self):
        # When both input sequences have the same structure, the output
        # sequence must have the same structure.
        self.assertEqual(
            all_possible_crossovers([0, 1, 2], [0, 1, 2]),
            {
                (0, 1, 2)
            })

    def test_crossover_uneven_length(self):
        # When the inputs have matching prefixes, that must be the same, but
        # either ending would be an acceptable crossover.
        self.assertEqual(
            all_possible_crossovers([0, 1], [0, 1, 2]),
            {
                (0, 1),
                (0, 1, 2)
            })

    def test_crossover_edge_alternatives(self):
        # In this case, both the beginning and the end of the sequence have
        # multiple possible variations, but the order must be consistent.
        self.assertEqual(
            all_possible_crossovers([0, 1], [1, 2]),
            {
                (0, 1),
                (1,),
                (1, 2),
                (0, 1, 2)
            })

    def test_crossover_divergent_at_end(self):
        # When the sequences differ at one position, either option is
        # acceptable, but the rest of the sequence must match.
        self.assertEqual(
            all_possible_crossovers([0, 1, 2], [0, 1, 3]),
            {
                (0, 1, 2),
                (0, 1, 3),
            })

    def test_crossover_divergent_at_beginning(self):
        # When the sequences differ at one position, either option is
        # acceptable, but the rest of the sequence must match. This covers an
        # edge case in the algorithm: an operation that only appears in one of
        # the inputs has a 50% chance of being selected. Each one is considered
        # independently, so care is taken not to reject BOTH possible values.
        # If that happened at the beginning of the list, the result would
        # always be an empty list, which is invalid output.
        self.assertEqual(
            all_possible_crossovers([2, 0, 1], [3, 0, 1]),
            {
                (2, 0, 1),
                (3, 0, 1),
            })

    def test_crossover_multiple_divergence_points(self):
        # When the sequences differ at more than one position, the result
        # should consider crossover with and without both single-parent values.
        self.assertEqual(
            all_possible_crossovers([0, 1, 2], [0, 3, 1]),
            {
                (0, 1),
                (0, 1, 2),
                (0, 3, 1),
                (0, 3, 1, 2),
            })

    def test_crossover_one_divergent_operation(self):
        # When the sequences differ at more than one position, the result
        # should consider crossover with and without both single-parent values.
        self.assertEqual(
            all_possible_crossovers([0], [1]),
            {
                (0,),
                (1,),
            })

    def test_crossover_all_divergent_operations(self):
        # When the sequences differ at more than one position, the result
        # should consider crossover with and without both single-parent values.
        self.assertEqual(
            all_possible_crossovers([0, 1, 2], [3, 4, 5]),
            {
                (0, 1, 2),
                (3, 4, 5),
            })

    def test_initial_population(self):
        constraints = Constraints(True, True, True)
        clade = Clade(experiments.NUM_SPECIES, constraints).serialize()
        progenitor = clade[0]
        diffs_per_species = []
        for species in clade[1:]:
            diffs_per_species.append(count_diffs(species, progenitor))
        num_mutants = np.count_nonzero(diffs_per_species)
        # 80% of the population is different from the progenitor.
        self.assertProportional(
            0.8 * experiments.NUM_SPECIES, num_mutants, delta=0.05)
        # On average, each species has two mutations.
        self.assertAlmostEqual(np.mean(diffs_per_species), 2.0, delta=0.1)
        # No species has more than 6 mutations.
        self.assertEqual(np.max(diffs_per_species), 6)

    def test_constraints_allow_composition(self):
        innovation_counter = itertools.count()
        program = PhenotypeProgram()
        program.add_draw(innovation_counter)
        constraints = Constraints(allow_composition=True)
        program.mutate(innovation_counter, DUMMY_GENOTYPE, constraints, 1.0)
        self.assertGreater(len(program.draw_ops), 1)

    def test_constraints_disallow_composition(self):
        innovation_counter = itertools.count()
        program = PhenotypeProgram()
        program.add_draw(innovation_counter)
        constraints = Constraints(allow_composition=False)
        program.mutate(innovation_counter, DUMMY_GENOTYPE, constraints, 1.0)
        self.assertEqual(len(program.draw_ops), 1)

    def test_constraints_allow_stamp_transforms(self):
        innovation_counter = itertools.count()
        program = PhenotypeProgram()
        program.add_draw(innovation_counter)
        constraints = Constraints(allow_stamp_transforms=True)
        program.mutate(innovation_counter, DUMMY_GENOTYPE, constraints, 1.0)
        num_stamp_transforms = 0
        for draw_op in program.draw_ops:
            num_stamp_transforms += len(draw_op.stamp_transforms)
        self.assertGreater(num_stamp_transforms, 0)

    def test_constraints_disallow_stamp_transforms(self):
        innovation_counter = itertools.count()
        program = PhenotypeProgram()
        program.add_draw(innovation_counter)
        constraints = Constraints(allow_stamp_transforms=False)
        program.mutate(innovation_counter, DUMMY_GENOTYPE, constraints, 1.0)
        num_stamp_transforms = 0
        for draw_op in program.draw_ops:
            num_stamp_transforms += len(draw_op.stamp_transforms)
        self.assertEqual(num_stamp_transforms, 0)

    def test_constraints_allow_bias(self):
        innovation_counter = itertools.count()
        program = PhenotypeProgram()
        program.add_draw(innovation_counter)
        constraints = Constraints(allow_bias=True)
        program.mutate(innovation_counter, DUMMY_GENOTYPE, constraints, 1.0)
        stamp_biases = 0
        transform_biases = 0
        for draw_op in program.draw_ops:
            stamp_biases += draw_op.stamp.bias_mode != BiasMode.NONE
            for transform in draw_op.global_transforms:
                transform_biases += sum(
                    arg.bias_mode != BiasMode.NONE
                    for arg in transform.args)
            for transform in draw_op.stamp_transforms:
                transform_biases += sum(
                    arg.bias_mode != BiasMode.NONE
                    for arg in transform.args)
        self.assertGreater(stamp_biases, 0)
        self.assertGreater(transform_biases, 0)

    def test_constraints_disallow_bias(self):
        innovation_counter = itertools.count()
        program = PhenotypeProgram()
        program.add_draw(innovation_counter)
        constraints = Constraints(allow_bias=False)
        program.mutate(innovation_counter, DUMMY_GENOTYPE, constraints, 1.0)
        stamp_biases = 0
        transform_biases = 0
        for draw_op in program.draw_ops:
            stamp_biases += draw_op.stamp.bias_mode != BiasMode.NONE
            for transform in draw_op.global_transforms:
                transform_biases += sum(
                    arg.bias_mode != BiasMode.NONE
                    for arg in transform.args)
            for transform in draw_op.stamp_transforms:
                transform_biases += sum(
                    arg.bias_mode != BiasMode.NONE
                    for arg in transform.args)
        self.assertEqual(stamp_biases, 0)
        self.assertEqual(transform_biases, 0)


if __name__ == '__main__':
    unittest.main()

