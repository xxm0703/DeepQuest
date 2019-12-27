import gym, time
env = gym.make('Seaquest-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(200):
        env.render()
        action = 8
        observation, reward, done, info = env.step(action)
        time.sleep(0.15)
        print(observation[50])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
