import time

import gym
import matplotlib.pyplot as plt
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb, [0.2989, 0.5870, 0.1140])


env = gym.make('Seaquest-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(200):
        env.render()
        action = 8
        observation, reward, done, info = env.step(action)
        gray = rgb2gray(observation)
        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=256)
        time.sleep(0.15)
        print(gray[50])
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        plt.show()
env.close()
