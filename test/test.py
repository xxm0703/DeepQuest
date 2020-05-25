import unittest
import gym
import numpy


def rgb2gray(rgb):
    # A proven formula for converting RGB to Gray-scale
    return numpy.dot(rgb, [0.2989, 0.5870, 0.1140])


class TestCases(unittest.TestCase):
    def test_red_to_grayscale(self):
        self.assertEqual(int(rgb2gray([255, 0, 0])), 76)

    def test_green_to_grayscale(self):
        self.assertEqual(int(rgb2gray([0, 255, 0])), 149)

    def test_blue_to_grayscale(self):
        self.assertEqual(int(rgb2gray([0, 0, 255])), 29)

    def test_action_space(self):
        env = gym.make('Seaquest-v0')
        self.assertEqual(env.action_space, gym.spaces.Discrete(18))

    def test_observation_space(self):
        env = gym.make('Seaquest-v0')
        self.assertEqual(env.observation_space.shape, (210, 160, 3))


if __name__ == '__main__':
    unittest.main()
