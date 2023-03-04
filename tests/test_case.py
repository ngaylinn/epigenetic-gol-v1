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

    def assertArrayEqual(self, a1, a2):
        if np.array_equal(a1, a2):
            return
        print('Arguments do not match.')
        print(f'First:  {a1.shape} == {a1}')
        print(f'Second: {a2.shape} == {a2}')

    def assertImagesEqual(self, image1, image2):
        """A debug method for comparing two images or videos."""
        if np.array_equal(image1, image2):
            return
        fig = plt.figure('Arguments do not match.')
        fig.add_subplot(1, 2, 1)
        a1 = gif_files.add_image_to_figure(image1, fig)
        fig.add_subplot(1, 2, 2)
        a2 = gif_files.add_image_to_figure(image2, fig)
        plt.show()

    def assertAllImagesEqual(self, image_list, prototype=None):
        if prototype is None:
            prototype = image_list.pop()
        for other_image in image_list:
            self.assertImagesEqual(prototype, other_image)

    def assertGolden(self, data):
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
        """
        class_name, test_name = self.id().split('.')[-2:]
        path = f'tests/{snake_case(class_name)}/'
        filename = f'{path}/{test_name}.gif'
        if os.path.exists(filename):
            golden_data = gif_files.load_image(filename)
            if np.array_equal(data, golden_data):
                return
            # At this point, a golden image was found and it doesn't match the
            # argument. Display a side by side to inspect the difference.
            message = f'{test_name}: Argument does not match golden file.'
            fig = plt.figure(message)
            axis = fig.add_subplot(1, 2, 1)
            axis.set_title('argument')
            a1 = gif_files.add_image_to_figure(data, fig)
            axis = fig.add_subplot(1, 2, 2)
            axis.set_title('golden')
            a2 = gif_files.add_image_to_figure(golden_data, fig)
            plt.show()
            self.fail(message)
        else:
            # If the golden file wasn't found, the directory for golden files
            # might not even be set up yet, so make sure it exists.
            os.makedirs(path, exist_ok=True)
            fig = plt.figure(f'{test_name}: Please manually verify.')
            gif_files.save_image(data, filename)
            animation = gif_files.add_image_to_figure(data, fig)
            plt.show()
            self.fail('No golden file found, so the argument has been saved'
                      'as the new golden file. Please validate and delete the'
                      'file if it is not correct before rerunning this test.')
