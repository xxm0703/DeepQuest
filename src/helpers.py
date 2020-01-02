import matplotlib.pyplot as plt
import numpy as np

__all__ = {
    'rgb2gray'
}


def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])


class LossPlotter:
    STEP = 0

    def __init__(self, x_axis='Time', y_axis='Loss', title='Real-time loss plot'):
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.title = title

    def setup(self):
        plt.title(self.title)
        plt.ylabel(self.y_axis)
        plt.xlabel(self.x_axis)

    def plot_loss(self, value):
        plt.plot(self.STEP, value, 'r+')
        plt.pause(1)
        plt.show()


a = LossPlotter()
a.plot_loss(13)
a.plot_loss(10)
