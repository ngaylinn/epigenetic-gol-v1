import unittest

import numpy as np

from kernel import simulate_organism, Cell, FitnessGoal, Simulator, WORLD_SIZE
from phenotype_program import TestClade
from tests import test_case


def count_alive(array):
    return np.count_nonzero(array == int(Cell.ALIVE))


class TestFitness(test_case.TestCase):
    def evolve_organism(self, goal):
        # The population size was chosen to be as small as possible while
        # producing stereotypic results from all fitness functions.
        simulator = Simulator(1, 1, 1200)
        clade = TestClade()
        fitness = simulator.evolve(clade.serialize(), goal, 100).flatten()
        genotypes = simulator.get_genotypes().flatten()
        best_genotype = genotypes[fitness.argmax()]
        video = simulate_organism(clade[0].serialize(), best_genotype)
        return fitness.max(), video

    def test_explode(self):
        fitness, video = self.evolve_organism(FitnessGoal.EXPLODE)
        alive_on_first = count_alive(video[0])
        alive_on_last = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * alive_on_last) // (1 + alive_on_first))
        self.assertGolden(video)

    def test_gliders(self):
        def same_value(array, value):
            if len(array) > 2:
                return np.logical_and(
                    array[0] == value, same_value(array[1:], value))
            else:
                return np.logical_and(
                    array[0] == value, array[1] == value)
        fitness, video = self.evolve_organism(FitnessGoal.GLIDERS)
        ROW_DELTA = 1
        COL_DELTA = 1
        TIME_DELTA = 4
        # Compare the last frame to the one TIME_DELTA frames before, but
        # shifted by ROW_DELTA rows and COL_DELTA cols. Cells in common
        # represent a pattern that is "moving" across the GOL world in the same
        # speed and direction as a glider.
        last_cycle = video[-TIME_DELTA - 1]
        this_cycle = np.pad(
            video[-1],
            ((0, ROW_DELTA), (0, COL_DELTA)),
            constant_values=Cell.DEAD
        )[-WORLD_SIZE:, -WORLD_SIZE:]
        in_spaceship = np.count_nonzero(
            np.logical_and(
                np.logical_and(last_cycle == int(Cell.ALIVE),
                               this_cycle == int(Cell.ALIVE)),
                np.logical_not(same_value(video[-TIME_DELTA:],
                                          int(Cell.ALIVE)))))
        not_in_spaceship = np.count_nonzero(
            np.logical_and(
                np.logical_or(last_cycle == int(Cell.DEAD),
                              this_cycle == int(Cell.DEAD)),
                np.logical_not(same_value(video[-TIME_DELTA:],
                                          int(Cell.DEAD)))))
        self.assertGolden(video)
        self.assertEqual(
            fitness, (100 * in_spaceship) // (1 + not_in_spaceship))

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
        # Count up cells where the last two steps match the two before, but the
        # cell changed states.
        cycling = np.count_nonzero(np.logical_and(repeating, moving))
        # Cells that were dead in both of the last two steps.
        dead = np.logical_and(
            last_cycle[-1] == int(Cell.DEAD),
            np.logical_and(
                last_cycle[-2] == int(Cell.DEAD),
                last_cycle[-3] == int(Cell.DEAD)))
        # Count up cells where the cell was alive at some point in the last two
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
        # Count up cells where the last two steps match the two before, but the
        # cell changed states.
        cycling = np.count_nonzero(np.logical_and(repeat, moving))
        # Cells that were dead in both of the last two steps.
        dead = np.logical_and(
            last_cycle[-1] == int(Cell.DEAD),
            last_cycle[-2] == int(Cell.DEAD))
        # Count up cells where the cell was alive at some point in the last two
        # steps but did not contribute to a repeating pattern.
        not_cycling = np.count_nonzero(
            np.logical_and(np.logical_not(repeat), np.logical_not(dead)))
        self.assertEqual(
            fitness, (100 * cycling) // (1 + not_cycling))
        self.assertGolden(video)


if __name__ == '__main__':
    unittest.main()
