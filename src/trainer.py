import gym

from helpers import rgb2gray, LossPlotter
from src.core import DQNAgent

EPISODES = 1000

if __name__ == "__main__":
    env = gym.make('Seaquest-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    plot = LossPlotter()
    agent.load("./save/seaquest-dqn.h5")
    K_frames = 4
    action = 0

    done = False
    batch_size = 32
    for e in range(EPISODES):
        state = env.reset()
        state = rgb2gray(state)
        for time in range(1000):
            if time % K_frames:
                action = agent.act(state)
                # env.render()
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10

            next_state = rgb2gray(next_state)  # Converting RGB state to gray-scale

            agent.memorize(state, action, reward, next_state, done)  # Remember

            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2f}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if time % 100 == 0:
                    plot.plot_loss(loss)

        if e % 2 == 0:
            print(f"Episode: {e}")
            agent.save("./save/seaquest-dqn.h5")
