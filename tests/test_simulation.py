"""Tests for kernel/gol_simulation.cu

These tests are meant to document and provide basic validation of the process
of running Game of Life simulations, capturing fitness scores and videos.
"""

import statistics
import time
import unittest

import numpy as np

import kernel
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

    def get_video_list(self, simulator):
        """Flatten all species and trials into a 1D list of videos."""
        videos = np.reshape(
            simulator.get_videos(),
            (simulator.size,
             kernel.NUM_STEPS, kernel.WORLD_SIZE, kernel.WORLD_SIZE))
        return [video for video in videos]

    def test_performance_benchmark(self):
        """Run time is about the same as previously observed."""
        num_trials = 3
        runtimes = []

        # Sometimes the GPU takes a moment to warm up, so use the median of
        # three trials to measure runtime.
        for _ in range(num_trials):
            start = time.perf_counter()
            simulator = kernel.TestSimulator(5, 5, 32)
            simulator.populate()
            for generation in range(200):
                simulator.simulate(kernel.FitnessGoal.STILL_LIFE)
                if generation + 1 < 200:
                    simulator.propagate()
            runtimes.append(time.perf_counter() - start)
        # The expected_runtime value is arbitrary, and different values will be
        # appropriate for different hardware. The purpose of this test is to
        # detect a sudden large change in performance from whenever this value
        # was last set, not to measure performance in any absolute terms.
        actual_runtime = statistics.median(runtimes)
        expected_runtime = 0.56
        delta = 0.01
        self.assertAlmostEqual(
            actual_runtime, expected_runtime, delta=delta)

    def test_reproducibility(self):
        """The same seed always produces the same simulated results."""
        num_trials = 3

        goal = kernel.FitnessGoal.STILL_LIFE
        simulator = kernel.TestSimulator(5, 5, 32)
        simulator.populate()
        simulator.simulate(goal, record=True)
        prototype_videos = self.get_video_list(simulator)
        prototype_fitness = simulator.get_fitness_scores()

        # After resetting the RNGs, should get the same result every time.
        for _ in range(num_trials - 1):
            simulator.reset_state()
            simulator.populate()
            simulator.simulate(goal, record=True)
            other_videos = self.get_video_list(simulator)
            for (prototype, other) in zip(prototype_videos, other_videos):
                self.assertImagesEqual(prototype, other)
            self.assertArrayEqual(
                prototype_fitness, simulator.get_fitness_scores())

    def test_game_of_life(self):
        """A Game of Life simulation proceeds according to the rules."""
        demo = np.full((kernel.WORLD_SIZE, kernel.WORLD_SIZE),
                       0xFF, dtype=np.uint8)
        demo[32:44, 32:44] = PINWHEEL
        demo[16:19, 16:19] = GLIDER

        simulator = kernel.TestSimulator(5, 5, 32)
        simulator.populate()
        simulator.simulate_phenotype(
            demo, kernel.FitnessGoal.STILL_LIFE, record=True)
        all_videos = self.get_video_list(simulator)
        self.assertAllImagesEqual(all_videos)
        self.assertGolden(all_videos[0])


if __name__ == '__main__':
    unittest.main()
