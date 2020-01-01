import numpy as np

__all__ = {
    'rgb2gray'
}


def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])
