import matplotlib.pyplot as plt
import numpy as np

__all__ = {
    'rgb2gray',
    'encapsulator',
}

from matplotlib.animation import FuncAnimation


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
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
        self.line, = self.ax.plot([], [], lw=3)
        self.setup()

    def setup(self):
        plt.title(self.title)
        plt.ylabel(self.y_axis)
        plt.xlabel(self.x_axis)

    def init_plot(self):
        self.line.set_data([], [])
        return self.line

    def start(self):
        FuncAnimation(self.fig, self.plot_loss, init_func=self.init_plot,
                      interval=200, blit=True)
        plt.show()

    def plot_loss(self, value):
        x = value
        y = np.sin(value * 0.1)
        self.line.set_data(x, y)
        return self.line
