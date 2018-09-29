import argparse
import random

import gym
import numpy as np
import skimage.color
import skimage.transform

from atari_games import Breakout
from dqn_agent import DQNAgent


def preprocess_frame(frame):
    gray_frame = skimage.color.rgb2gray(frame)
    resized_frame = skimage.transform.resize(gray_frame, [84, 84])
    # Store on smallest type possible (uint8) because replay experience will
    # take huge amount of memory, hence using float is very costly
    compact_frame = np.uint8(resized_frame * 255)
    return compact_frame


def train(game):
    env = gym.make(game.env_name)
    agent = DQNAgent()
    nb_steps = 0

    for _ in range(agent.nb_episodes):
        done, terminal = False, False
        score, lives = game.start_score, game.start_lives
        observation = env.reset()

        # To avoid sub-optimal, start the episode by waiting for a few steps
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
            game_action = game.get_ingame_action(action)
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
            if nb_steps % agent.target_update_rate == 0:
                agent.update_target_model()

            # Prepare for next iteration
            if terminal:
                terminal = False
            else:
                history = next_history
            nb_steps += 1


def play(game, model_path):
    print(model_path)
    pass


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", action="store_true",
                    help="train the agent to play the game")
parser.add_argument("-p", "--play", type=str, metavar="MODEL_FILE",
                    help="load agent model file to play the game")
args = parser.parse_args()

game = Breakout()
if args.train:
    train(game)
elif args.play:
    play(game, args.play)
else:
    parser.print_help()
