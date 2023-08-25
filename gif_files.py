"""Save, load, and display GOL simulation videos as gif files.

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


def simulation_data_to_image(frame):
    """Create a single Image from a 2D Numpy array."""
    scale = IMAGE_SCALE_FACTOR
    resized = frame.repeat(scale, 0).repeat(scale, 1)
    image = Image.fromarray(resized, mode='L')
    return image


def save_simulation_data_as_image(data, filename):
    """Save an image or video to a file.

    Parameters
    ----------
    data : np.ndarray of np.uint8
        A 2D Numpy array representing a still image or a 3D array representing
        a video.
    filename : a string describing where to save the file.
    """
    if data.ndim == 2:
        simulation_data_to_image(data).save(filename)
        return
    assert data.ndim == 3
    images = [simulation_data_to_image(frame) for frame in data]
    durations = [MILLISECONDS_FOR_PHENOTYPE]
    durations.extend([MILLISECONDS_PER_FRAME] * (len(images) - 1))
    images[0].save(
        filename, save_all=True, append_images=images[1:], loop=0,
        duration=durations)


def image_to_simulation_data(image):
    """Create a 2D Numpy array from a single Image."""
    raw_data = np.array(image.convert('L'), dtype=np.uint8)
    return raw_data[::IMAGE_SCALE_FACTOR, ::IMAGE_SCALE_FACTOR]


def load_simulation_data_from_image(filename):
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
            return image_to_simulation_data(image)
        frames = []
        for frame in range(n_frames):
            image.seek(frame)
            # When PIL exports gif files, it merges adjacent frames that are
            # equivalent and just increases the duration of the static frame.
            # That's a problem, since we expect all our videos to have the same
            # number of frames, and there's no way to disable this behavior.
            # To restore the original frames, then, we must calculate how many
            # frames got merged using the known duration per frame.
            if frame > 0:
                repeats = image.info['duration'] // MILLISECONDS_PER_FRAME
            else:
                repeats = 1
            for _ in range(repeats):
                frames.append(image_to_simulation_data(image))
        return np.array(frames)


def add_simulation_data_to_figure(data, fig, axis):
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
    axis.spines[:].set_visible(True)
    plt.setp(axis.spines.values(), color='#ff0000')
    plt.setp(axis.spines.values(), linewidth=1)

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


def display_simulation_data(data):
    """Pop up a window visualizing simulation data as an image / video."""
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    anim = add_simulation_data_to_figure(data, fig, axis)
    plt.show()
