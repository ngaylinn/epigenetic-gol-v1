"""Tests for kernel/reproduction.h

These tests are meant to document behavior and provide basic validation.
"""

import random
import unittest

import matplotlib.pyplot as plt
import numpy as np

import gif_files
from kernel import (
    breed_population,
    Cell, GenotypeDType, Simulator,
    CELLS_PER_STAMP, CROSSOVER_RATE, MUTATION_RATE, NUM_GENES, STAMP_SIZE)
from phenotype_program import TestClade
from tests import test_case


def visualize_genotype(genotype):
    """Render a Genotype as a plt figure with images."""
    cols = NUM_GENES
    fig = plt.figure("Genotype")
    for gene_index in range(cols):
        axis = fig.add_subplot(2, cols, gene_index + 1)
        # Make a primitive bar chart representing the Scalar genes
        # that's the same width and appearance as a Stamp gene value.
        scale_factor = (1 << 32) / STAMP_SIZE
        raw_value = genotype['scalar_genes'][gene_index]
        scaled_value = int(raw_value / scale_factor)
        scalar_viz = np.pad(
            np.full((2, scaled_value), 0x00, dtype=np.uint8),
            ((0, 0), (0, STAMP_SIZE - scaled_value)),
            constant_values=0xFF)
        gif_files.add_simulation_data_to_figure(scalar_viz, fig, axis)
        axis.set_title(f'{100 * raw_value / (0xFFFFFFFF):0.2f}%')
    for gene_index in range(cols):
        axis = fig.add_subplot(2, cols, cols + gene_index + 1)
        gif_files.add_simulation_data_to_figure(
            genotype['stamp_genes'][gene_index], fig, axis)


class TestReproduction(test_case.TestCase):
    """Tests for initializing and breeding populations of Genotypes."""

    def test_randomize(self):
        """Random genes have the expected distributional properties."""
        num_species, num_trials, num_organisms = 5, 5, 32
        simulator = Simulator(num_species, num_trials, num_organisms)
        clade = TestClade(5)
        simulator.populate(clade.serialize())
        genotypes = simulator.get_genotypes()
        # For every trial, look at the genes of all the organisms in that
        # population and make sure they are randomized appropriately.
        for species_index in range(num_species):
            for trial_index in range(num_trials):
                organism_genotypes = genotypes[species_index][trial_index]
                scalar_values = organism_genotypes['scalar_genes'].flatten()
                stamp_values = organism_genotypes['stamp_genes'].flatten()
                msg = f'species {species_index}, trial {trial_index}'
                # Scalar genes are 50% of their max value, on average.
                self.assertProportional(
                    1 << 31, scalar_values.mean(), delta=0.11, msg=msg)
                # Scalar gene values are almost all unique, since few values
                # are being drawn from the full range of 32-bit ints.
                self.assertAlmostEqual(
                    num_organisms * NUM_GENES,
                    np.unique(scalar_values).size,
                    delta=0.01, msg=msg)
                # The average stamp value is halfway between ALIVE and DEAD.
                self.assertProportional(
                    (int(Cell.DEAD) + int(Cell.ALIVE)) / 2,
                    stamp_values.mean(),
                    0.11, msg=msg)
                # About half of the stamp values are ALIVE.
                self.assertProportional(
                    len(stamp_values) / 2,
                    np.count_nonzero(stamp_values == int(Cell.ALIVE)),
                    0.11, msg=msg)
                # All stamp values are either ALIVE or DEAD.
                self.assertEqual(
                    len(stamp_values),
                    np.count_nonzero(
                        np.logical_or(
                            stamp_values == int(Cell.ALIVE),
                            stamp_values == int(Cell.DEAD))))

    def test_sample_random_genotypes(self):
        """Collect visualizations of random genotypes to verify manually."""
        num_organisms = 8
        simulator = Simulator(1, 1, num_organisms)
        clade = TestClade(1)
        simulator.populate(clade.serialize())
        genotypes = simulator.get_genotypes()
        for organism_index in range(num_organisms):
            visualize_genotype(genotypes[0][0][organism_index])
            path, test_name = self.get_test_data_location()
            # SVG would be a better graphics format, but the pyplot library has
            # a bug where SVG file outputs are not deterministic.
            plt.savefig(f'{path}/{test_name}{organism_index}.png')
            plt.close()

    def test_reproducibility(self):
        """The same seed produces the same pseudorandom genotypes."""
        def single_trial():
            result = {}
            simulator = Simulator(5, 5, 32)
            clade = TestClade(5)
            simulator.populate(clade.serialize())
            result['before'] = simulator.get_genotypes()
            simulator.propagate()
            result['after'] = simulator.get_genotypes()
            return result
        num_trials = 3
        results = [single_trial() for _ in range(num_trials)]
        prototype = results.pop()
        for result in results:
            self.assertArrayEqual(
                prototype['before']['scalar_genes'],
                result['before']['scalar_genes'])
            self.assertArrayEqual(
                prototype['before']['stamp_genes'],
                result['before']['stamp_genes'])
            self.assertArrayEqual(
                prototype['after']['scalar_genes'],
                result['after']['scalar_genes'])
            self.assertArrayEqual(
                prototype['after']['stamp_genes'],
                result['after']['stamp_genes'])

    def test_mutations(self):
        """Genes mutate during reproduction as expected."""
        # Use a larger population size so we'll get enough mutations to
        # measure with some precision.
        num_species, num_trials, num_organisms = 32, 32, 32
        population_size = num_species * num_trials * num_organisms

        # Set all Genotype values to 0 and have every organism breed with
        # itself. Any non-zero values are the result of mutations.
        genotypes = np.zeros(
            (num_species, num_trials, num_organisms), dtype=GenotypeDType)
        parent_selections = list(range(population_size))
        mate_selections = parent_selections

        # Actually do the breeding
        genotypes = breed_population(
            genotypes, parent_selections, mate_selections)
        scalar_values = genotypes['scalar_genes'].flatten()
        stamp_values = genotypes['stamp_genes'].flatten()

        # Assert mutation rate is as expected.
        self.assertProportional(
            MUTATION_RATE * NUM_GENES * population_size,
            np.count_nonzero(scalar_values),
            delta=0.1)
        alive_probability = 0.5
        self.assertProportional(
            (MUTATION_RATE * NUM_GENES * CELLS_PER_STAMP *
             population_size * alive_probability),
            np.count_nonzero(stamp_values),
            delta=0.1)

    def test_crossover(self):
        """Reproduction uses crossover at the expected rate."""
        num_species, num_trials, num_organisms = 5, 5, 32
        half_organisms = int(num_organisms / 2)
        population_size = num_species * num_trials * num_organisms

        # Set up Genotypes and select organisms such that each parent has all
        # of its genes set to its max value and each mate has all its genes set
        # to a low value. Then we can see how many genes from parents and mates
        # make it through the breeding process.
        parent_selections = []
        mate_selections = []
        population_index = 0
        genotypes = np.empty(
            (num_species, num_trials, num_organisms), dtype=GenotypeDType)
        for species_index in range(num_species):
            for trial_index in range(num_trials):
                for organism_index in range(num_organisms):
                    genotype = (
                        genotypes[species_index][trial_index][organism_index])
                    if organism_index < half_organisms:
                        genotype['scalar_genes'].fill(0xFFFFFFFF)
                        genotype['stamp_genes'].fill(0xFF)
                        parent_selections.append(population_index)
                        mate_selections.append(
                            population_index + half_organisms)
                    else:
                        genotype['scalar_genes'].fill(0x00000000)
                        genotype['stamp_genes'].fill(0x00)
                        parent_selections.append(
                            population_index - half_organisms)
                        mate_selections.append(population_index)
                    population_index += 1

        # Actually do the breeding.
        genotypes = breed_population(
            genotypes, parent_selections, mate_selections)
        scalar_values = genotypes['scalar_genes'].flatten()
        stamp_values = genotypes['stamp_genes'].flatten()

        # Assert that genes got remixed as expected. Note, rather than
        # computing expected values that take mutation rates into account, we
        # just set slightly looser bounds. This works pretty well since the
        # crossover rate is much greater than the mutation rate.
        # If there was no crossover, all genes come from parent. If there was
        # crossover, 50% of genes come from parent.
        parent_gene_rate = (
            (1 - CROSSOVER_RATE) + 0.5 * CROSSOVER_RATE)
        self.assertProportional(
            parent_gene_rate * NUM_GENES * population_size,
            np.count_nonzero(scalar_values),
            delta=0.02)
        self.assertProportional(
            (parent_gene_rate * NUM_GENES * CELLS_PER_STAMP * population_size),
            np.count_nonzero(stamp_values),
            delta=0.02)


if __name__ == '__main__':
    unittest.main()
