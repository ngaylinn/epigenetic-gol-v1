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


def get_video_list(simulator):
    """Flatten all species and trials into a 1D list of videos."""
    videos = np.reshape(
        simulator.get_videos(),
        (simulator.size,
         kernel.NUM_STEPS, kernel.WORLD_SIZE, kernel.WORLD_SIZE))
    return videos.tolist()


class TestSimulation(test_case.TestCase):
    """Sanity checks for running Game of Life simulations."""

    def test_reproducibility(self):
        """The same seed always produces the same simulated results."""
        def single_trial():
            result = {}
            goal = kernel.FitnessGoal.STILL_LIFE
            simulator = kernel.Simulator(3, 3, 32)
            simulator.populate(phenotype_program.get_defaults(3))
            simulator.simulate(goal, record=True)
            result['videos'] = get_video_list(simulator)
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
        simulator.populate(phenotype_program.get_defaults(3))
        simulator.simulate(goal, record=True)
        gpu_video = get_video_list(simulator)[0]
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
