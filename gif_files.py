"""Save, load, and display simulation videos as gif files.

This project generates Game of Life simulations as Numpy arrays, but these get
serialized to disk and displayed in the form of gif images (either single
images or videos). This library takes care of translating the simulation data
format to and from gif files with the desired appearance.
"""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image


# When exporting a simulation video, scale it up by this much to make it
# easier to see.
IMAGE_SCALE_FACTOR = 2

# Controls the playback speed of the animated gif, including an extended
# duration for the first frame, so you can see the phenotype clearly.
MILLISECONDS_PER_FRAME = 100
MILLISECONDS_FOR_PHENOTYPE = 10 * MILLISECONDS_PER_FRAME


def make_image(frame):
    """Create a single Image from a 2D Numpy array."""
    scale = IMAGE_SCALE_FACTOR
    resized = frame.repeat(scale, 0).repeat(scale, 1)
    image = Image.fromarray(resized, mode='L')
    return image


def save_image(data, filename):
    """Save an image or video to a file.

    Parameters
    ----------
    data : np.ndarray of np.uint8
        A 2D Numpy array representing a still image or a 3D array representing
        a video.
    filename : a string describing where to save the file.
    """
    if data.ndim == 2:
        make_image(data).save(filename)
        return
    n_frames = data.shape[0]
    images = [make_image(frame) for frame in data]
    durations = [MILLISECONDS_FOR_PHENOTYPE]
    durations.extend([MILLISECONDS_PER_FRAME] * n_frames)
    images[0].save(
        filename, save_all=True, append_images=images[1:], loop=0,
        duration=durations)


def make_array(image):
    """Create a 2D Numpy array from a single Image."""
    raw_data = np.asarray(image, dtype=np.uint8)
    resized = raw_data[::IMAGE_SCALE_FACTOR, ::IMAGE_SCALE_FACTOR]
    normalized = (resized != 0x00) * 0xFF
    return normalized


def load_image(filename):
    """Load an image or video from a file.

    Parameters
    ----------
    filename : a string describing where to load the file from.

    Returns
    -------
    np.ndarray of np.uint8
        An array representing the image stored in filename. This may be a 2D
        array for a single image or a 3D array for a video.
    """
    with Image.open(filename) as image:
        n_frames = getattr(image, 'n_frames', 1)
        if n_frames == 1:
            return make_array(image)
        frames = []
        for frame in range(n_frames):
            image.seek(frame)
            # Get channel shouldn't be necessary, but for some reason videos I
            # save with all single-chanel frames load with the first frame
            # single-chanel and the remaining frames with three channels.
            frames.append(make_array(image.getchannel(0)))
        return np.array(frames)


def add_image_to_figure(data, fig):
    """Display an image in an existing PLT figure.

    Parameters
    ----------
    data : np.ndarray of np.uint8
        A 2D Numpy array representing a still image or a 3D array representing
        a video.
    fig : A PLT figure. This function will append to that figure using imshow.

    Returns
    -------
    If data is a still image, returns None. Otherwise, returns an animation
    object that the caller is responsible for holding onto until calling
    plt.show().
    """
    plt.axis('off')
    format_options = {
        'cmap': 'gray',
        'vmin': 0,
        'vmax': 255
    }
    if data.ndim == 2:
        plt.imshow(data, **format_options)
        return None

    image = plt.imshow(data[0], **format_options)

    def animate_func(i):
        image.set_array(data[i])
        return image

    anim = animation.FuncAnimation(
        fig, animate_func, frames=data.shape[0],
        interval=MILLISECONDS_PER_FRAME)
    return anim
