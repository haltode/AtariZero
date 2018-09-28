import random

import gym
import numpy as np
import skimage.color
import skimage.transform

from dqn_agent import DQNAgent


NB_EPISODES = 50000
GAME_NAME = 'BreakoutDeterministic-v4'


def preprocess_frame(frame):
    gray_frame = skimage.color.rgb2gray(frame)
    resized_frame = skimage.transform.resize(gray_frame, [84, 84])
    # Store on smallest type possible (uint8) because replay experience will
    # take huge amount of memory, hence using float is very costly
    compact_frame = np.uint8(resized_frame * 255)
    return compact_frame


def translate_to_game_action(action):
    if 'Breakout' in GAME_NAME:
        if action == 0:
            return 1
        elif action == 1:
            return 2
        else:
            return 3


def starting_situation():
    if 'Breakout' in GAME_NAME:
        return 0, 5


env = gym.make(GAME_NAME)
agent = DQNAgent()

for _ in range(NB_EPISODES):
    done, terminal = False, False
    score, lives = starting_situation()
    observation = env.reset()

    # To avoid sub-optimal, start the episode by doing nothing for a few steps
    nb_no_op = random.randint(1, agent.no_op_max_steps)
    for _ in range(nb_no_op):
        observation, _, _, _ = env.step(1)

    # Initial history
    state = preprocess_frame(observation)
    history = np.stack((state, state, state, state), axis=2)
    history = np.reshape([history], (1, 84, 84, 4))

    while not done:
        # Play action
        action = agent.choose_action(history)
        game_action = translate_to_game_action(action)
        observation, reward, done, info = env.step(game_action)

        # Update history
        next_state = preprocess_frame(observation)
        next_state = np.reshape([next_state], (1, 84, 84, 1))
        next_history = np.append(next_state, history[:, :, :, :3], axis=3)

        # Lost a life
        if lives > info['ale.lives']:
            terminal = True
            lives = info['ale.lives']

        reward = np.clip(reward, -1., 1.)
        score += reward

        # Learn
        agent.save_to_memory(history, action, reward, next_history, terminal)
        agent.train_replay()

        # Prepare for next iteration
        if terminal:
            terminal = False
        else:
            history = next_history
