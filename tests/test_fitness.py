import unittest

import numpy as np

import gif_files
import kernel
import phenotype_program
from tests import test_case


def count_alive(array):
    return np.count_nonzero(array == int(kernel.ALIVE))


class TestFitness(test_case.TestCase):
    def evolve_organism(self, goal):
        simulator = kernel.Simulator(1, 1, 1000)
        clade = phenotype_program.Clade(1, testing=True)
        # TODO: Switch to using the evolve method.
        # simulator.evolve(np.full((1), program), goal, 100)
        simulator.populate(clade.serialize())
        for _ in range(99):
            simulator.simulate(goal)
            simulator.propagate()
        simulator.simulate(goal)
        fitness = simulator.get_fitness_scores().flatten()
        genotypes = simulator.get_genotypes().flatten()
        best_genotype = genotypes[fitness.argmax()]
        video = kernel.simulate_organism(clade[0].serialize(), best_genotype)
        return fitness.max(), video

    def test_explode(self):
        fitness, video = self.evolve_organism(kernel.FitnessGoal.EXPLODE)
        self.assertGolden(video)
        alive_on_first = count_alive(video[0])
        alive_on_last = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * alive_on_last) // (1 + alive_on_first))

    def test_gliders(self):
        fitness, video = self.evolve_organism(kernel.FitnessGoal.GLIDERS)
        ROW_DELTA = 1
        COL_DELTA = 1
        TIME_DELTA = 4
        # Compare the last frame to the one TIME_DELTA frames before, but
        # shifted by ROW_DELTA rows and COL_DELTA cols. Cells in common
        # represent a pattern that is "moving" across the GOL world in the same
        # speed and direction as a glider.
        before = np.pad(
            video[-TIME_DELTA - 1],
            ((ROW_DELTA, 0), (COL_DELTA, 0)),
            constant_values=kernel.DEAD
        )[:kernel.WORLD_SIZE, :kernel.WORLD_SIZE]
        after = video[-1]
        repeating = np.count_nonzero(
            np.logical_and(before == int(kernel.ALIVE),
                           after == int(kernel.ALIVE)))
        live_cells = np.count_nonzero(video[-1] == int(kernel.ALIVE))
        self.assertGolden(video)
        self.assertEqual(
            fitness, (100 * repeating) // (1 + live_cells))

    def test_left_to_right(self):
        fitness, video = self.evolve_organism(kernel.FitnessGoal.LEFT_TO_RIGHT)
        self.assertGolden(video)
        first_frame_left, first_frame_right = np.split(video[0], 2, axis=1)
        last_frame_left, last_frame_right = np.split(video[-1], 2, axis=1)
        on_target_total = (
            np.count_nonzero(first_frame_left == int(kernel.ALIVE)) +
            np.count_nonzero(last_frame_right == int(kernel.ALIVE)))
        off_target_total = (
            np.count_nonzero(first_frame_right == int(kernel.ALIVE)) +
            np.count_nonzero(last_frame_left == int(kernel.ALIVE)))
        # TODO: Clean this up once you're done playing with this fitness goal.
        # targets = np.linspace(
        #     0, kernel.WORLD_SIZE - 1, kernel.NUM_STEPS, dtype=np.uint8)
        # window = kernel.WORLD_SIZE // 8
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

    def test_still_life(self):
        fitness, video = self.evolve_organism(kernel.FitnessGoal.STILL_LIFE)
        self.assertGolden(video)
        static_cells = np.count_nonzero(
            np.logical_and(
                video[-1] == int(kernel.ALIVE),
                video[-2] == video[-1]))
        live_cells = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * static_cells) // (1 + static_cells - live_cells))

    def test_symmetry(self):
        fitness, video = self.evolve_organism(kernel.FitnessGoal.SYMMETRY)
        self.assertGolden(video)
        alive = video[-1] == int(kernel.ALIVE)
        h_mirror = np.flip(video[-1], axis=1) == int(kernel.ALIVE)
        v_mirror = np.flip(video[-1], axis=0) == int(kernel.ALIVE)
        symmetries = (
            np.count_nonzero(np.logical_and(alive, h_mirror)) +
            np.count_nonzero(np.logical_and(alive, v_mirror)))
        assymmetries = (
            np.count_nonzero(np.logical_and(alive, np.logical_not(h_mirror))) +
            np.count_nonzero(np.logical_and(alive, np.logical_not(v_mirror))))
        self.assertEqual(
            fitness, (100 * symmetries) // (1 + assymmetries))

    def test_three_cycle(self):
        fitness, video = self.evolve_organism(kernel.FitnessGoal.THREE_CYCLE)
        self.assertGolden(video)
        prev_a, prev_b, prev_c, last_a, last_b, last_c = video[-6:]
        a_same = last_a == prev_a
        b_same = last_b == prev_b
        c_same = last_c == prev_c
        static = np.logical_and(last_a == last_b, last_b == last_c)
        cycling = np.count_nonzero(
            np.logical_and(
                a_same, np.logical_and(
                    b_same, np.logical_and(
                        c_same, np.logical_not(static)))))
        live_cells = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * cycling) // (1 + live_cells))

    def test_two_cycle(self):
        fitness, video = self.evolve_organism(kernel.FitnessGoal.TWO_CYCLE)
        self.assertGolden(video)
        prev_a, prev_b, last_a, last_b = video[-4:]
        a_same = last_a == prev_a
        b_same = last_b == prev_b
        static = last_a == last_b
        cycling = np.count_nonzero(
            np.logical_and(
                a_same, np.logical_and(
                    b_same, np.logical_not(static))))
        live_cells = count_alive(video[-1])
        self.assertEqual(
            fitness, (100 * cycling) // (1 + live_cells))


if __name__ == '__main__':
    unittest.main()
