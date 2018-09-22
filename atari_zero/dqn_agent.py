import collections
import random

import keras
import numpy as np


class DQNAgent:
    def __init__(self):
        self.state_size = (84, 84, 4)
        self.action_size = 3

        self.init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.nb_exploration_steps = 1000000
        self.epsilon_decay = ((self.init_epsilon - self.final_epsilon)
                              / self.nb_exploration_steps)

        self.epsilon = self.init_epsilon

        self.memory = collections.deque(maxlen=400000)
        self.batch_size = 32
        self.discount_rate = 0.99
        self.replay_start_size = 50000

        self.build_atari_model()

    def build_atari_model(self):
        frames_input = keras.layers.Input(self.state_size, name='frames')
        actions_input = keras.layers.Input((self.action_size, ), name='mask')

        normalized = keras.layers.Lambda(lambda x: x / 255.)(frames_input)

        layer1 = keras.layers.convolutional.Conv2D(
            16, (8, 8), strides=(4, 4), activation='relu'
        )(normalized)
        layer2 = keras.layers.convolutional.Conv2D(
            32, (4, 4), strides=(2, 2), activation='relu'
        )(layer1)
        flat_layer2 = keras.layers.core.Flatten()(layer2)
        layer3 = keras.layers.Dense(256, activation='relu')(flat_layer2)

        output = keras.layers.Dense(self.action_size)(layer3)
        filtered_output = keras.layers.Multiply(name="Q_values")(
            [output, actions_input]
        )

        self.model = keras.models.Model(
            inputs=[frames_input, actions_input], outputs=filtered_output
        )
        self.optimizer = keras.optimizers.RMSprop(
            lr=0.00025, rho=0.95, epsilon=0.01
        )
        self.model.compile(self.optimizer, loss='mse')

    def fit_single_batch(self, gamma, actions, rewards,
                         start_states, next_states, is_terminal):
        # Predict Q values of the next states
        next_Q_values = self.model.predict(
            [next_states, np.ones(actions.shape)]
        )
        next_Q_values[is_terminal] = 0

        # Calculate Q values of start states
        Q_values = rewards + gamma * np.max(next_Q_values, axis=1)

        # Fit Keras model
        training_data = [start_states, actions]
        target_data = actions * Q_values[:, None]
        self.model.fit(
            training_data, target_data, epochs=1,
            batch_size=len(start_states), verbose=0
        )

    def choose_action(self, history):
        history = np.float32(history / 255.)
        # epsilon greedy exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def save_to_memory(self, history, action, reward, next_history, terminal):
        self.memory.append((history, action, reward, next_history, terminal))

    def train_replay(self):
        if len(self.memory) < self.replay_start_size:
            return
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        action, reward, terminal = [], [], []
        target = np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            terminal.append(mini_batch[i][4])

        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            target[i] = reward[i]
            if not terminal[i]:
                target[i] += self.discount_rate * np.amax(target_value[i])

        action_one_hot = np.eye(self.action_size)[np.array(action).reshape(-1)]
        training_data = [history, action_one_hot]
        target_data = action_one_hot * target[:, None]
        self.model.fit(
            training_data, target_data, epochs=1,
            batch_size=self.batch_size, verbose=0
        )
