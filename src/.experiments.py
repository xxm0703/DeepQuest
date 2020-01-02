import time

import gym
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.models import Sequential

from src.helpers import rgb2gray


def cnn3dilated(input_shape):
    model = Sequential(name='cnn3adam')
    model.add(Conv2D(kernel_size=5, filters=32, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    print(model.output_shape)

    model.add(Conv2D(kernel_size=3, filters=32, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    print(model.output_shape)

    model.add(Conv2D(kernel_size=3, filters=32, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    print(model.output_shape)

    model.add(Dropout(0.2))
    model.add(Flatten())
    print(model.output_shape)


def gray_scale():
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


def dimentions():
    env = gym.make('Seaquest-v0')
    print(f"Input dim: {env.action_space.n}")
    print(f"Output dim: {env.observation_space.shape}")


if __name__ == '__main__':
    dimentions()
    cnn3dilated((210, 160, 1))
