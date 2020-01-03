import gym

from helpers import rgb2gray
from src.core import DQNAgent

EPISODES = 1000

if __name__ == "__main__":
    env = gym.make('Seaquest-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/seaquest-dqn.h5")
    done = False
    batch_size = 32
    i = 0
    for e in range(EPISODES):
        state = env.reset()
        state = rgb2gray(state)
        for time in range(500):
            i += 1
            env.render()
            action = agent.act(state)
            next_state, reward, done, hint = env.step(action)
            reward = reward if not done else -10
            print(hint)
            next_state = rgb2gray(next_state)  # Converting RGB state to gray-scale

            agent.memorize(state, action, reward, next_state, done)  # Remember
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("./save/seaquest-dqn.h5")
