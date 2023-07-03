"""Tests for kernel/gol_simulation.cu

These tests are meant to document behavior and provide basic validation.
"""

import unittest

import numpy as np

import kernel
import phenotype_program
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
            goal = kernel.FitnessGoal.STILL_LIFE
            simulator = kernel.Simulator(3, 3, 32)
            clade = phenotype_program.Clade(3, testing=True)
            simulator.populate(clade.serialize())
            videos = simulator.simulate_and_record(goal)
            # Flatten out the collection of videos orgnanized by species,
            # trial, and organism into a flat list with one video for each
            # individual in the population.
            result['videos'] = np.reshape(
                videos,
                (simulator.size,
                 kernel.NUM_STEPS, kernel.WORLD_SIZE, kernel.WORLD_SIZE))
            result['fitness'] = simulator.get_fitness_scores()
            return result
        num_trials = 3
        results = [single_trial() for _ in range(num_trials)]
        prototype = results.pop()
        for result in results:
            for (video1, video2) in zip(prototype['videos'], result['videos']):
                self.assertImagesEqual(video1, video2)
            self.assertArrayEqual(prototype['fitness'], result['fitness'])

    def test_gpu_and_cpu_agree(self):
        """The fancy GPU-optimized simulation matches the basic one."""
        goal = kernel.FitnessGoal.STILL_LIFE
        simulator = kernel.Simulator(3, 3, 32)
        clade = phenotype_program.Clade(3, testing=True)
        simulator.populate(clade.serialize())
        # Grab just the first video from the GPU (the rest should be the same)
        gpu_video = simulator.simulate_and_record(goal)[0][0][0]
        # Run the CPU simulation with the same randomly generated phenotype we
        # used on the GPU, but then recompute the rest of the video.
        cpu_video = kernel.simulate_phenotype(gpu_video[0])
        self.assertImagesEqual(gpu_video, cpu_video)

    def test_game_of_life(self):
        """A Game of Life simulation proceeds according to the rules."""
        demo = np.full((kernel.WORLD_SIZE, kernel.WORLD_SIZE),
                       0xFF, dtype=np.uint8)
        demo[32:44, 32:44] = PINWHEEL
        demo[16:19, 16:19] = GLIDER

        self.assertGolden(kernel.simulate_phenotype(demo))


if __name__ == '__main__':
    unittest.main()
