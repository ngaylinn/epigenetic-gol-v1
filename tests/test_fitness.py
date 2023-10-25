import unittest

import numpy as np

from kernel import simulate_organism, Cell, FitnessGoal, NUM_STEPS, WORLD_SIZE
from evolution import (
    NUM_ORGANISM_GENERATIONS, TestClade)
from tests import test_case


def count_alive(array):
    return np.count_nonzero(array == int(Cell.ALIVE))


class TestFitness(test_case.TestCase):
    def evolve_organism(self, goal):
        clade = TestClade()
        # Evolve organisms and capture video of the final generation. Flatten
        # the population to a single dimension, but keep each of the Videos in
        # their characteristic shape.
        # TODO: Is it better to just record the whole last generation, or to
        # use simulate organism on just the best one? Do we still need
        # kernel.simulate_organism?
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

    def test_explode(self):
        fitness, video = self.evolve_organism(FitnessGoal.EXPLODE)
        alive_on_first = count_alive(video[0])
        alive_on_last = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * alive_on_last) // (1 + alive_on_first))
        self.assertGolden(video)

    # TODO: Add missing FitnessGoals!

    def test_left_to_right(self):
        fitness, video = self.evolve_organism(FitnessGoal.LEFT_TO_RIGHT)
        first_frame_left, first_frame_right = np.split(video[0], 2, axis=1)
        last_frame_left, last_frame_right = np.split(video[-1], 2, axis=1)
        on_target_total = (
            np.count_nonzero(first_frame_left == int(Cell.ALIVE)) +
            np.count_nonzero(last_frame_right == int(Cell.ALIVE)))
        off_target_total = (
            np.count_nonzero(first_frame_right == int(Cell.ALIVE)) +
            np.count_nonzero(last_frame_left == int(Cell.ALIVE)))
        # TODO: Clean this up once you're done playing with this fitness goal.
        # targets = np.linspace(
        #     0, WORLD_SIZE - 1, NUM_STEPS, dtype=np.uint8)
        # window = WORLD_SIZE // 8
        # on_target_total = 0
        # off_target_total = 0
        # for step, target in enumerate(targets):
        #     frame = video[step]
        #     min_col = target - window + 1
        #     max_col = target + window
        #     on_target = count_alive(frame[..., min_col:max_col])
        #     alive = count_alive(frame)
        #     off_target = alive - on_target
        #     on_target_total += on_target
        #     off_target_total += off_target
        self.assertEqual(
            fitness, (100 * on_target_total) // (1 + off_target_total))
        self.assertGolden(video)

    def test_still_life(self):
        fitness, video = self.evolve_organism(FitnessGoal.STILL_LIFE)
        static_cells = np.count_nonzero(
            np.logical_and(
                video[-1] == int(Cell.ALIVE),
                video[-2] == video[-1]))
        live_cells = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * static_cells) // (1 + static_cells - live_cells))
        self.assertGolden(video)

    def test_symmetry(self):
        fitness, video = self.evolve_organism(FitnessGoal.SYMMETRY)
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
        self.assertGolden(video)

    def test_three_cycle(self):
        fitness, video = self.evolve_organism(FitnessGoal.THREE_CYCLE)
        last_cycle = video[-3:]
        prev_cycle = video[-6:-3]
        # Cells where the last two steps match the previous two
        repeating = np.logical_and(
            last_cycle[-1] == prev_cycle[-1],
            np.logical_and(
                last_cycle[-2] == prev_cycle[-2],
                last_cycle[-3] == prev_cycle[-3]))
        # Cells where the last two steps had the same value
        moving = np.logical_or(
            last_cycle[-1] != last_cycle[-2],
            last_cycle[-2] != last_cycle[-3])
        # Count up Cells where the last two steps match the two before, but the
        # Cell changed states.
        cycling = np.count_nonzero(np.logical_and(repeating, moving))
        # Cells that were DEAD in both of the last two steps.
        dead = np.logical_and(
            last_cycle[-1] == int(Cell.DEAD),
            np.logical_and(
                last_cycle[-2] == int(Cell.DEAD),
                last_cycle[-3] == int(Cell.DEAD)))
        # Count up Cells where the Cell was ALIVE at some point in the last two
        # steps but did not contribute to a repeating pattern.
        not_cycling = np.count_nonzero(
            np.logical_and(np.logical_not(repeating), np.logical_not(dead)))
        self.assertEqual(
            fitness, (100 * cycling) // (1 + not_cycling))
        self.assertGolden(video)

    def test_two_cycle(self):
        fitness, video = self.evolve_organism(FitnessGoal.TWO_CYCLE)
        last_cycle = video[-2:]
        prev_cycle = video[-4:-2]
        # Cells where the last two steps match the previous two
        repeat = np.logical_and(
            last_cycle[-1] == prev_cycle[-1],
            last_cycle[-2] == prev_cycle[-2])
        # Cells where the last two steps had the same value
        moving = last_cycle[-1] != last_cycle[-2]
        # Count up Cells where the last two steps match the two before, but the
        # cell changed states.
        cycling = np.count_nonzero(np.logical_and(repeat, moving))
        # Cells that were DEAD in both of the last two steps.
        dead = np.logical_and(
            last_cycle[-1] == int(Cell.DEAD),
            last_cycle[-2] == int(Cell.DEAD))
        # Count up Cells where the Cell was ALIVE at some point in the last two
        # steps but did not contribute to a repeating pattern.
        not_cycling = np.count_nonzero(
            np.logical_and(np.logical_not(repeat), np.logical_not(dead)))
        self.assertEqual(
            fitness, (100 * cycling) // (1 + not_cycling))
        self.assertGolden(video)


if __name__ == '__main__':
    unittest.main()
