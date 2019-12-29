import gym
import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])


env = gym.make('Seaquest-v0')
observation = env.reset()
for _ in range(20):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    gray = rgb2gray(observation)
    plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=256)
plt.show()
env.close()
