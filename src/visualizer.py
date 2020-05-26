import time

import gym

from core import DQNAgent
from helpers import rgb2gray

EPISODES = 5000

if __name__ == "__main__":
    env = gym.make('Seaquest-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/seaquest-dqn-save.h5")
    done = False
    batch_size = 32
    K_frames = 3
    action = 0
    i = 0
    while True:
        state = env.reset()
        state = rgb2gray(state)
        i += 1
        for t in range(4000):
            if t % K_frames == 0:
                action = agent.decide(state)
            isOpened = env.render()

            if not isOpened:
                env.close()
                exit(0)

            next_state, reward, done, _ = env.step(action)

            state = rgb2gray(next_state)
            if done:
                print("episode: {}, score: {}"
                      .format(i, t))
                break
            time.sleep(0.05)
