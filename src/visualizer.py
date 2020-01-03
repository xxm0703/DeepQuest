import gym

from helpers import rgb2gray
from src.core import DQNAgent

EPISODES = 5000

if __name__ == "__main__":
    env = gym.make('Seaquest-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/seaquest-dqn.h5")
    done = False
    batch_size = 32
    i = 0
    while True:
        state = env.reset()
        state = rgb2gray(state)
        i += 1
        for time in range(1500):
            env.render()
            action = agent.decide(state)
            next_state, reward, done, _ = env.step(action)

            next_state = rgb2gray(next_state)  # Converting RGB state to gray-scale

            state = next_state
            if done:
                print("episode: {}, score: {}, e: {:.2}"
                      .format(i, time, agent.epsilon))
                break
