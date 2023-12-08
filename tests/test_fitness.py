import unittest

import numpy as np

from kernel import Cell, FitnessGoal, NUM_STEPS, WORLD_SIZE
from evolution import (
    NUM_ORGANISM_GENERATIONS, TestClade)
from tests import test_case


def count_alive(array):
    return np.count_nonzero(array == int(Cell.ALIVE))


def cycling_fitness(cycle_length, video):
    """Compute fitness for cycling simulations."""
    # Look at just the last few frames...
    NUM_ITERATIONS = 4
    video = video[-NUM_ITERATIONS*cycle_length:]

    # Count up cells that stayed on or off the whole time.
    always_on = np.full((WORLD_SIZE, WORLD_SIZE), True)
    always_off = np.full((WORLD_SIZE, WORLD_SIZE), True)
    for frame in video:
        always_on = np.logical_and(always_on, frame == int(Cell.ALIVE))
        always_off = np.logical_and(always_off, frame == int(Cell.DEAD))

    # Split the video into CYCLE_LENGTH chunks and count up how many cells
    # repeated themselves at that frequency.
    iterations = np.array(np.split(video, NUM_ITERATIONS))
    repeating = np.full((WORLD_SIZE, WORLD_SIZE), True)
    for i in range(1, NUM_ITERATIONS):
        repeating = np.logical_and(
            repeating,
            np.all(iterations[0] == iterations[i], axis=0))

    # Count up cells that are repeating but not static.
    cycling = np.count_nonzero(np.logical_and(
        np.logical_not(always_on),
        np.logical_and(
            np.logical_not(always_off),
            repeating)))

    # Count up cells that were on at some point, but aren't repeating.
    not_cycling = np.count_nonzero(np.logical_and(
        np.logical_not(always_off),
        np.logical_not(repeating)))

    # Compute a final fitness score.
    return (cycling * cycling) / (1 + not_cycling)


class TestFitness(test_case.TestCase):
    def evolve_organism(self, goal):
        clade = TestClade()
        # Evolve organisms and capture video of the final generation. Flatten
        # the population to a single dimension, but keep each of the Videos in
        # their characteristic shape.
        simulations = clade.evolve_organisms(goal, True)
        simulations = simulations.reshape(
            -1, NUM_STEPS, WORLD_SIZE, WORLD_SIZE)
        # Flatten the population to a single dimensions, and look at just the
        # last generation of fitness scores.
        fitness = clade.organism_fitness_history
        fitness = fitness.reshape(-1, NUM_ORGANISM_GENERATIONS)[:, -1]
        # Pick out the simulation Video corresponding to the most fit
        # organism in the population.
        best_simulation = simulations[fitness.argmax()]
        assert best_simulation.shape == (NUM_STEPS, WORLD_SIZE, WORLD_SIZE)
        return fitness.max(), best_simulation

    def test_entropy(self):
        _, video = self.evolve_organism(FitnessGoal.ENTROPY)
        # For entropy, just collect an example video. Unlike the other
        # FitnessGoals, there's no good way to validate the logic with a
        # Python-based implementation, since it depends on the particulars of
        # the nvcomp compression algorithms.
        self.assertGolden(video)

    def test_explode(self):
        fitness, video = self.evolve_organism(FitnessGoal.EXPLODE)
        alive_on_first = count_alive(video[0])
        alive_on_last = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * alive_on_last) // (1 + alive_on_first))
        self.assertGolden(video)

    def test_left_to_right(self):
        fitness, video = self.evolve_organism(FitnessGoal.LEFT_TO_RIGHT)
        self.assertGolden(video)
        first_frame_left, first_frame_right = np.split(video[0], 2, axis=1)
        last_frame_left, last_frame_right = np.split(video[-1], 2, axis=1)
        on_target_first_frame = (
            np.count_nonzero(first_frame_left == int(Cell.ALIVE)) +
            np.count_nonzero(first_frame_right == int(Cell.DEAD)))
        on_target_last_frame = (
            np.count_nonzero(last_frame_left == int(Cell.DEAD)) +
            np.count_nonzero(last_frame_right == int(Cell.ALIVE)))
        self.assertEqual(
            fitness, on_target_first_frame + 4 * on_target_last_frame)

    def test_ring(self):
        fitness, video = self.evolve_organism(FitnessGoal.RING)
        self.assertGolden(video)
        last_frame = video[-1]
        on_target_count = 0
        off_target_count = 0
        CENTER = WORLD_SIZE // 2
        INNER_RADIUS = CENTER // 4
        OUTER_RADIUS = 3 * CENTER // 4
        for row in range(WORLD_SIZE):
            for col in range(WORLD_SIZE):
                distance_squared = (CENTER - row) ** 2 + (CENTER - col) ** 2
                within_target = (distance_squared > INNER_RADIUS ** 2 and
                                 distance_squared <= OUTER_RADIUS ** 2)
                if within_target == (last_frame[row][col] == int(Cell.ALIVE)):
                    on_target_count += 1
                else:
                    off_target_count += 1
        self.assertEqual(
            fitness, int((100 * on_target_count) / (1 + off_target_count)))

    def test_still_life(self):
        fitness, video = self.evolve_organism(FitnessGoal.STILL_LIFE)
        self.assertGolden(video)
        static_cells = np.count_nonzero(
            np.logical_and(
                video[-1] == int(Cell.ALIVE),
                video[-2] == video[-1]))
        live_cells = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * static_cells) // (1 + live_cells - static_cells))

    def test_symmetry(self):
        fitness, video = self.evolve_organism(FitnessGoal.SYMMETRY)
        self.assertGolden(video)
        alive = video[-1] == int(Cell.ALIVE)
        h_mirror = np.flip(video[-1], axis=1) == int(Cell.ALIVE)
        v_mirror = np.flip(video[-1], axis=0) == int(Cell.ALIVE)
        symmetries = (
            np.count_nonzero(np.logical_and(alive, h_mirror)) +
            np.count_nonzero(np.logical_and(alive, v_mirror)))
        assymmetries = (
            np.count_nonzero(np.logical_and(alive, np.logical_not(h_mirror))) +
            np.count_nonzero(np.logical_and(alive, np.logical_not(v_mirror))))
        self.assertEqual(
            fitness, (100 * symmetries) // (1 + assymmetries))

    def test_three_cycle(self):
        fitness, video = self.evolve_organism(FitnessGoal.THREE_CYCLE)
        self.assertGolden(video)
        self.assertEqual(fitness, cycling_fitness(3, video))

    def test_two_cycle(self):
        fitness, video = self.evolve_organism(FitnessGoal.TWO_CYCLE)
        self.assertGolden(video)
        self.assertEqual(fitness, cycling_fitness(2, video))


if __name__ == '__main__':
    unittest.main()
