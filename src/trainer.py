import time

import gym

from core import DQNAgent
from helpers import rgb2gray, LossPlotter, encapsulator

EPISODES = 1000

if __name__ == "__main__":
    env = gym.make('Seaquest-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    plot = LossPlotter()
    agent.load("./save/seaquest-dqn3.h5")
    K_frames = 3
    action = 0
    stop_watch = time.time()

    done = False
    batch_size = 16
    for e in range(EPISODES):
        state = env.reset()
        state = rgb2gray(state)
        for frame in range(1000):
            if frame % K_frames == 0:
                action = agent.act(state)
            # env.render()
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -50

            next_state = rgb2gray(next_state)  # Converting RGB state to gray-scale

            # agent.memorize(encapsulator(state), action, reward, encapsulator(next_state), done)  # Remember
            if reward != 0 or frame % 100 == 0:
                agent.memorize(encapsulator(state), action, reward, encapsulator(next_state), done)  # Remember

            state = next_state
            if done:
                print("episode: {}/{}, frame: {}, e: {:.2f}"
                      .format(e, EPISODES, frame, agent.epsilon))
                break
            if len(agent.memory) > batch_size and frame % 10 == 0:
                loss = agent.replay(batch_size)
                print("episode: {}/{}, time: {:.2f}, loss: {:.4f}"
                      .format(e, EPISODES, time.time() - stop_watch, loss))
                stop_watch = time.time()

        if e % 2 == 0:
            print(f"Episode: {e}")
            agent.save("./save/seaquest-dqn3.h5")
