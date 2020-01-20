import matplotlib.pyplot as plt
import numpy as np

__all__ = {
    'rgb2gray',
    'encapsulator',
    'LossPlotter',
}


def rgb2gray(rgb):
    # A proven formula for converting RGB to Gray-scale
    gray_frame = np.dot(rgb, [0.2989, 0.5870, 0.1140])
    # Put the color in a container, to simulate a color-channel
    return gray_frame.reshape(210, 160, 1)


def encapsulator(frame):
    """
    Accepts a frame and encapsulates it into a np.array.
    Now when this array is passed to model.predict,
    it simulates that it is a batch of frames with length of 1
    """
    return np.array((frame,))


class LossPlotter:
    STEP = 0

    def __init__(self, x_axis='Time', y_axis='Loss', title='Real-time loss plot'):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.title = title
        self.setup()

    def setup(self):
        plt.title(self.title)
        plt.ylabel(self.y_axis)
        plt.xlabel(self.x_axis)

    def plot_loss(self, value):
        plt.plot(self.STEP, value, 'r+')
        plt.pause(0.001)
        self.STEP += 1
