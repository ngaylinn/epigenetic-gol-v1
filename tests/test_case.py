import os.path
import unittest

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gif_files


def snake_case(camel_case_string):
    return ''.join(['_' + char.lower() if char.isupper() else char
                    for char in camel_case_string]).lstrip('_')


class TestCase(unittest.TestCase):
    """A base TestCase for this project with useful assertions."""

    def get_test_data_location(self):
        """Generate a per-test directory name for golden file data."""
        class_name, test_name = self.id().split('.')[-2:]
        path = f'tests/{snake_case(class_name)}/'
        os.makedirs(path, exist_ok=True)
        return path, test_name

    def assertProportional(self, expected, actual, delta, msg=None):
        """Assert that second is within some multiple of first."""
        difference = abs(expected - actual)
        percent = difference / expected
        self.assertLessEqual(
            percent, delta,
            f'difference between {expected} and {actual} is '
            f'{100 * percent:0.2f}%, which is not less than '
            f'{100 * delta:0.2f}%: {msg}')

    def assertArrayEqual(self, a1, a2, msg=None):
        """Assert two Numpy arrays are exactly equal."""
        self.assertEqual(
            a1.size, a2.size,
            f'Arrays are different sizes ({a1.size} vs {a2.size}): {msg}')
        self.assertEqual(
            a1.shape, a2.shape,
            f'Arrays are different shapes ({a1.shape} vs {a2.shape}): {msg}')
        diffs = np.count_nonzero(a1 - a2)
        size = a1.size
        self.assertEqual(
            0, diffs,
            f'Arrays differ in {diffs} of {size} positions: {msg}')

    def assertSimulationEqual(self, simulation1, simulation2):
        """Assert that two simulation frames or videos are the same."""
        if np.array_equal(simulation1, simulation2):
            return
        fig = plt.figure('Arguments do not match.')
        axis = fig.add_subplot(1, 2, 1)
        a1 = gif_files.add_simulation_data_to_figure(simulation1, fig, axis)
        axis = fig.add_subplot(1, 2, 2)
        a2 = gif_files.add_simulation_data_to_figure(simulation2, fig, axis)
        plt.show()

    def assertGolden(self, data, test_id=None):
        """Verify that frame matches output from a previous run.

        This assertion checks to see if the output from a previous run of this
        test has already been saved as a file (in which case it is presumed to
        have been reviewed and approved by a person). If it has, then the
        assertion checks that the current output matches that "golden" output
        from before. If no golden file is present, this assertion will generate
        one for the developer to review.

        Parameters
        ----------
        data : np.ndarray of np.uint8
            An array representing image data. This can either be a 2D array
            representing a single image, or a 3D array representing a video
            with several frames. All pixels are assumed to be grayscale values
            between 0 and 255.
        test_id : string
            By default, this method uses the test name as the name of the
            golden file. To call assertGolden more than once in a single test,
            this string will be appended to differentiate the files.
        """
        path, test_name = self.get_test_data_location()
        if test_id is not None:
            test_name = f'{test_name}_{test_id}'
        filename = f'{path}/{test_name}.gif'
        if os.path.exists(filename):
            golden_data = gif_files.load_simulation_data_from_image(filename)
            if np.array_equal(data, golden_data):
                return
            # At this point, a golden image was found and it doesn't match the
            # argument. Display a side by side to inspect the difference.
            message = f'{test_name}: Argument does not match golden file.'
            fig = plt.figure(message)
            axis = fig.add_subplot(1, 2, 1)
            axis.set_title('argument')
            a1 = gif_files.add_simulation_data_to_figure(data, fig, axis)
            axis = fig.add_subplot(1, 2, 2)
            axis.set_title('golden')
            a2 = gif_files.add_simulation_data_to_figure(golden_data, fig, axis)
            plt.show()
            self.fail(message)
        else:
            # If the golden file wasn't found, the directory for golden files
            # might not even be set up yet, so make sure it exists.
            gif_files.save_simulation_data_as_image(data, filename)
            fig = plt.figure(f'{test_name}: Please manually verify.')
            axis = fig.add_subplot(1, 1, 1)
            animation = gif_files.add_simulation_data_to_figure(data, fig, axis)
            plt.show()
            print('No golden file found, so the argument has been saved '
                  'as the new golden file. Please validate and delete the '
                  'file if it is not correct before rerunning this test.')
