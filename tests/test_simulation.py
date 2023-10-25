"""Tests for kernel/gol_simulation.cu

These tests are meant to document behavior and provide basic validation.
"""

import unittest

import numpy as np

from kernel import (
    simulate_phenotype,
    FitnessGoal, Simulator,
    NUM_STEPS, WORLD_SIZE)
from evolution import TestClade, NUM_ORGANISM_GENERATIONS
from tests import test_case

GLIDER = np.array(
    [[0xFF, 0x00, 0xFF],
     [0xFF, 0xFF, 0x00],
     [0x00, 0x00, 0x00]],
    dtype=np.uint8)

PINWHEEL = np.array(
    [[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
     [0x00, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0x00, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00],
     [0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x00],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]],
    dtype=np.uint8)


class TestSimulation(test_case.TestCase):
    """Sanity checks for running Game of Life simulations."""

    def test_reproducibility(self):
        """The same seed always produces the same simulated results."""
        def single_trial():
            result = {}
            goal = FitnessGoal.STILL_LIFE
            clade = TestClade()
            simulations = clade.evolve_organisms(goal, True)
            # Flatten out the collection of Videos orgnanized by species,
            # trial, and organism into a flat list with one Video for each
            # individual in the population.
            result['sims'] = simulations.reshape(
                (-1, NUM_STEPS, WORLD_SIZE, WORLD_SIZE))
            result['fitness'] = clade.organism_fitness_history.reshape(
                (-1, NUM_ORGANISM_GENERATIONS))[:, -1]
            return result
        num_trials = 3
        results = [single_trial() for _ in range(num_trials)]
        prototype = results.pop()
        for result in results:
            for (sim1, sim2) in zip(prototype['sims'], result['sims']):
                self.assertSimulationEqual(sim1, sim2)
            self.assertArrayEqual(prototype['fitness'], result['fitness'])

    def test_gpu_and_cpu_agree(self):
        """The fancy GPU-optimized simulation matches the basic one."""
        goal = FitnessGoal.STILL_LIFE
        clade = TestClade()
        # Grab just the first Video from the GPU (the rest should be the same)
        gpu_simulation = clade.evolve_organisms(goal, True)[0][0][0]
        # Run the CPU simulation with the same randomly generated phenotype we
        # used on the GPU, but then recompute the rest of the Video.
        cpu_simulation = simulate_phenotype(gpu_simulation[0])
        self.assertSimulationEqual(gpu_simulation, cpu_simulation)

    def test_game_of_life(self):
        """A Game of Life simulation proceeds according to the rules."""
        demo = np.full((WORLD_SIZE, WORLD_SIZE), 0xFF, dtype=np.uint8)
        demo[32:44, 32:44] = PINWHEEL
        demo[16:19, 16:19] = GLIDER

        self.assertGolden(simulate_phenotype(demo))


if __name__ == '__main__':
    unittest.main()
