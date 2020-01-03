import random
from collections import deque

import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from helpers import encapsulator


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size[:2] + (1,)  # no. inputs + 1-channel color
        self.action_size = action_size  # no. outputs
        self.memory = deque(maxlen=2000)  # decision register
        self.gamma = 0.8  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.02
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential(name='Questor1.0')
        model.add(Conv2D(kernel_size=5, filters=8, input_shape=self.state_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(kernel_size=3, filters=16, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Conv2D(kernel_size=3, filters=8, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))

        model.add(Dropout(0.2))
        model.add(Flatten())

        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return self.decide(state)

    def decide(self, state):
        state = encapsulator(state)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state = encapsulator(next_state)
                a = self.model.predict(next_state)
                target = (reward + self.gamma * np.amax(a[0]))

            state = encapsulator(state)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, verbose=False)  # verbose=0

            loss = history.history['loss'][0]

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def plot(self, loss):
        pass
