
import unittest

import numpy as np

import kernel
from kernel import OperationType
from phenotype_program import PhenotypeProgram
from tests import test_case

# Just a simple asymmetrical pattern with clear extents, useful for visualizing
# the various coordinate transformations used in development.
DEMO_STAMP = np.array(
    [[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00],
     [0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x00],
     [0x00, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00],
     [0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0xFF, 0x00],
     [0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00],
     [0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00],
     [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]],
    dtype=np.uint8)


def test_op(op_type, op_args):
    program = PhenotypeProgram()
    program.add_operation(op_type, op_args)
    program.add_operation(OperationType.TEST)
    return kernel.render_phenotype(program.data)


class TestDevelopment(test_case.TestCase):
    def test_array_1d(self):
        self.assertGolden(test_op(OperationType.ARRAY_1D, (13, 25)))

    def test_array_2d(self):
        self.assertGolden(test_op(OperationType.ARRAY_2D, (25, 13)))

    def test_copy(self):
        self.assertGolden(test_op(OperationType.COPY, (13, 25)))

    def test_crop(self):
        self.assertGolden(test_op(OperationType.CROP, (13, 25)))

    def test_flip(self):
        self.assertGolden(test_op(OperationType.FLIP, 0), 'none')
        self.assertGolden(test_op(OperationType.FLIP, 1), 'vertical')
        self.assertGolden(test_op(OperationType.FLIP, 2), 'horizontal')
        self.assertGolden(test_op(OperationType.FLIP, 3), 'both')

    def test_mask(self):
        self.assertGolden(test_op(OperationType.MASK, DEMO_STAMP))

    def test_mirror(self):
        self.assertGolden(test_op(OperationType.MIRROR, 0), 'none')
        self.assertGolden(test_op(OperationType.MIRROR, 1), 'vertical')
        self.assertGolden(test_op(OperationType.MIRROR, 2), 'horizontal')
        self.assertGolden(test_op(OperationType.MIRROR, 3), 'both')

    def test_quarter(self):
        self.assertGolden(test_op(OperationType.QUARTER, 0b0001), 'up_left')
        self.assertGolden(test_op(OperationType.QUARTER, 0b0110), 'checker')
        self.assertGolden(test_op(OperationType.QUARTER, 0b0011), 'left_half')

    def test_rotate(self):
        self.assertGolden(test_op(OperationType.ROTATE, 0), '0_degrees')
        self.assertGolden(test_op(OperationType.ROTATE, 1), '90_degrees')
        self.assertGolden(test_op(OperationType.ROTATE, 2), '180_degrees')
        self.assertGolden(test_op(OperationType.ROTATE, 3), '270_degrees')

    def test_scale(self):
        self.assertGolden(test_op(OperationType.SCALE, (1, 1)), '1x')
        self.assertGolden(test_op(OperationType.SCALE, (2, 2)), '2x')
        self.assertGolden(test_op(OperationType.SCALE, (8, 1)), 'stretch')

    def test_tile(self):
        self.assertGolden(test_op(OperationType.TILE, (0, 0)), 'aligned')
        self.assertGolden(test_op(OperationType.TILE, (4, 0)), 'staggered')
        self.assertGolden(test_op(OperationType.TILE, (0, 1)), 'alternating')
        self.assertGolden(test_op(OperationType.TILE, (4, 1)),
                          'staggered_alternating')

    def test_translate(self):
        self.assertGolden(test_op(OperationType.TRANSLATE, (13, 25)))


if __name__ == '__main__':
    unittest.main()
