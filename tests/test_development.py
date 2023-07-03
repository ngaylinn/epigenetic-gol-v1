import itertools
import unittest

import numpy as np

import kernel
from kernel import TransformType
import phenotype_program
from tests import test_case

# Just a simple asymmetrical pattern with clear extents, useful for visualizing
# the various coordinate transformations used in development.
ORIENTED_STAMP = np.array(
    [[0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
     [0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00],
     [0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00, 0x00],
     [0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00],
     [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]],
    dtype=np.uint8)

# Stamps showing the numbers 1-4 to test using multiple stamps
STAMP_1 = np.array(
    [[0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
     [0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF],
     [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00],
     [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]],
    dtype=np.uint8)

STAMP_2 = np.array(
    [[0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
     [0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00],
     [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF],
     [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00],
     [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]],
    dtype=np.uint8)

STAMP_3 = np.array(
    [[0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
     [0x00, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00],
     [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF],
     [0xFF, 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF],
     [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00],
     [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]],
    dtype=np.uint8)

STAMP_4 = np.array(
    [[0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00],
     [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0x00],
     [0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF],
     [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF],
     [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00],
     [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00]],
    dtype=np.uint8)

# A patch of black, used for testing composition with bitwise operations
FULL_STAMP = np.array(
    [[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]],
    dtype=np.uint8)


def run_test_program(prototype):
    innovation_counter = itertools.count()
    program = phenotype_program.PhenotypeProgram()
    for (stamp, compose, global_transforms, stamp_transforms) in prototype:
        draw_op = program.add_draw(innovation_counter)
        draw_op.compose_mode = compose
        draw_op.stamp.bias_mode = kernel.BiasMode.FIXED_VALUE
        draw_op.stamp.bias = stamp

        for (transform_type, transform_args) in global_transforms:
            transform = draw_op.add_global_transform(innovation_counter)
            transform.type = transform_type
            for index, arg in enumerate(transform_args):
                transform.args[index].bias_mode = kernel.BiasMode.FIXED_VALUE
                transform.args[index].bias = arg

        for (transform_type, transform_args) in stamp_transforms:
            transform = draw_op.add_stamp_transform(innovation_counter)
            transform.type = transform_type
            for index, arg in enumerate(transform_args):
                transform.args[index].bias_mode = kernel.BiasMode.FIXED_VALUE
                transform.args[index].bias = arg
    return kernel.render_phenotype(program.serialize())


class TestDevelopment(test_case.TestCase):
    def test_global_transform_array_1d(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_1D, (13, 25))],
                 [])
            ]))

    def test_global_transform_array_2d(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (13, 25))],
                 [])
            ]))

    def test_global_transform_copy(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.COPY, (13, 25))],
                 [])
            ]))

    def test_global_transform_crop(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.CROP, (13, 25)),
                  (TransformType.TILE, (0, 0))],
                 [])
            ]))

    def test_global_transform_flip(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.FLIP, (0,))],
                 [])
            ]), 'none')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.FLIP, (1,))],
                 [])
            ]), 'vertical')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.FLIP, (2,))],
                 [])
            ]), 'horizontal')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.FLIP, (3,))],
                 [])
            ]), 'both')

    def test_global_transform_mirror(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.MIRROR, (0,))],
                 [])
            ]), 'none')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.MIRROR, (1,))],
                 [])
            ]), 'vertical')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.MIRROR, (2,))],
                 [])
            ]), 'horizontal')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.MIRROR, (3,))],
                 [])
            ]), 'both')

    def test_global_transform_quarter(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.QUARTER, (0b0001,)),
                  (TransformType.TILE, (0, 0))],
                 [])
            ]), 'up_left')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.QUARTER, (0b0110,)),
                  (TransformType.TILE, (0, 0))],
                 [])
            ]), 'checker')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.QUARTER, (0b0011,)),
                  (TransformType.TILE, (0, 0))],
                 [])
            ]), 'left_half')

    def test_global_transform_rotate(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ROTATE, (0,))],
                 [])
            ]), '0_degrees')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ROTATE, (1,))],
                 [])
            ]), '90_degrees')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ROTATE, (2,))],
                 [])
            ]), '180_degrees')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ROTATE, (3,))],
                 [])
            ]), '270_degrees')

    def test_global_transform_scale(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.SCALE, (1, 1))],
                 [])
            ]), '1x')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.SCALE, (2, 2))],
                 [])
            ]), '2x')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.SCALE, (8, 1))],
                 [])
            ]), 'stretch')

    def test_global_transform_tile(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.TILE, (0, 0))],
                 [])
            ]), 'align')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.TILE, (4, 0))],
                 [])
            ]), 'stagger')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.TILE, (0, 1))],
                 [])
            ]), 'alternate')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.TILE, (4, 1))],
                 [])
            ]), 'stagger_alternate')

    def test_global_transform_translate(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (13, 25))],
                 [])
            ]))

    def test_stamp_transform_crop(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.CROP, (6, 6))])
            ]))

    def test_stamp_transform_flip(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.FLIP, (0,))])
            ]), 'none')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.FLIP, (1,))])
            ]), 'vertical')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.FLIP, (2,))])
            ]), 'horizontal')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.FLIP, (3,))])
            ]), 'both')

    def test_stamp_transform_mirror(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.MIRROR, (0,))])
            ]), 'none')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.MIRROR, (1,))])
            ]), 'vertical')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.MIRROR, (2,))])
            ]), 'horizontal')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.MIRROR, (3,))])
            ]), 'both')

    def test_stamp_transform_quarter(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.QUARTER, (0b0001,))])
            ]), 'up_left')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.QUARTER, (0b0110,))])
            ]), 'checker')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.QUARTER, (0b0011,))])
            ]), 'left_half')

    def test_stamp_transform_rotate(self):
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.ROTATE, (0,))])
            ]), '0_degrees')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.ROTATE, (1,))])
            ]), '90_degrees')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.ROTATE, (2,))])
            ]), '180_degrees')
        self.assertGolden(
            run_test_program([
                (ORIENTED_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 [(TransformType.ROTATE, (3,))])
            ]), '270_degrees')

    def test_compose_two_draws(self):
        self.assertGolden(
            run_test_program([
                (STAMP_1, kernel.ComposeMode.OR,
                 [],
                 []),
                (STAMP_2, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (0, 8))],
                 [])
            ]))

    def test_compose_four_draws(self):
        self.assertGolden(
            run_test_program([
                (STAMP_1, kernel.ComposeMode.OR,
                 [],
                 []),
                (STAMP_2, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (0, 8))],
                 []),
                (STAMP_3, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (8, 8))],
                 []),
                (STAMP_4, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (8, 0))],
                 [])
            ]))

    def test_compose_same_stamp(self):
        self.assertGolden(
            run_test_program([
                (STAMP_1, kernel.ComposeMode.OR,
                 [],
                 []),
                (STAMP_1, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (0, 8))],
                 []),
                (STAMP_1, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (8, 8))],
                 []),
                (STAMP_1, kernel.ComposeMode.OR,
                 [(TransformType.TRANSLATE, (8, 0))],
                 [])
            ]))

    def test_compose_modes(self):
        self.assertGolden(
            run_test_program([
                (STAMP_1, kernel.ComposeMode.OR,
                 [(TransformType.QUARTER, (0b0101,)),
                  (TransformType.ARRAY_2D, (8, 8))],
                 []),
                (FULL_STAMP, kernel.ComposeMode.NONE,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 []),
            ]), 'none')
        self.assertGolden(
            run_test_program([
                (STAMP_1, kernel.ComposeMode.OR,
                 [(TransformType.QUARTER, (0b0101,)),
                  (TransformType.ARRAY_2D, (8, 8))],
                 []),
                (FULL_STAMP, kernel.ComposeMode.OR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 []),
            ]), 'or')
        self.assertGolden(
            run_test_program([
                (STAMP_1, kernel.ComposeMode.OR,
                 [(TransformType.QUARTER, (0b0101,)),
                  (TransformType.ARRAY_2D, (8, 8))],
                 []),
                (FULL_STAMP, kernel.ComposeMode.XOR,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 []),
            ]), 'xor')
        self.assertGolden(
            run_test_program([
                (STAMP_1, kernel.ComposeMode.OR,
                 [(TransformType.QUARTER, (0b0101,)),
                  (TransformType.ARRAY_2D, (8, 8))],
                 []),
                (FULL_STAMP, kernel.ComposeMode.AND,
                 [(TransformType.ARRAY_2D, (16, 16))],
                 []),
            ]), 'and')


if __name__ == '__main__':
    unittest.main()
