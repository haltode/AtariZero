import argparse
import datetime
import os
import random
import time

import gym
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf

from atari_games import Breakout, Pong
from dqn_agent import DQNAgent


def preprocess_frame(frame):
    gray_frame = skimage.color.rgb2gray(frame)
    resized_frame = skimage.transform.resize(gray_frame, [84, 84], mode='constant')
    # Store on smallest type possible (uint8) because replay experience will
    # take huge amount of memory, hence using float is very costly
    compact_frame = np.uint8(resized_frame * 255)
    return compact_frame


def get_ingame_action(action):
    return action + 1


def train(env, game, model_path):
    agent = DQNAgent()
    nb_steps = 0

    current_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join("log", current_date)
    writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    for episode in range(agent.nb_episodes):
        done, terminal = False, False
        score, lives = game.start_score, game.start_lives
        loss = 0.
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
            game_action = get_ingame_action(action)
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

            # Learn
            agent.save_to_memory(history, action, reward, next_history, terminal)

            score += reward
            loss += agent.train_replay()

            if nb_steps % agent.target_update_rate == 0:
                agent.update_target_model()

            # Prepare for next iteration
            if terminal:
                terminal = False
            else:
                history = next_history
            nb_steps += 1

        print("done episode {}: loss {}, score {}.".format(episode, loss, score))
        if episode % 100 == 0:
            agent.model.save(model_path)

        # Output log into TensorBoard
        loss_summary = tf.Summary(
            value=[tf.Summary.Value(tag="loss", simple_value=loss)])
        score_summary = tf.Summary(
            value=[tf.Summary.Value(tag="score", simple_value=score)])
        writer.add_summary(loss_summary, episode)
        writer.add_summary(score_summary, episode)

    agent.model.save(model_path)
    writer.close()


def play(env, game, model_path):
    agent = DQNAgent(model_path)

    done, score = False, game.start_score
    observation = env.reset()

    # Initial history
    state = preprocess_frame(observation)
    history = np.stack((state, state, state, state), axis=2)
    history = np.reshape([history], (1, 84, 84, 4))

    while not done:
        env.render()
        time.sleep(0.05)

        # Play action
        action = agent.choose_action(history)
        game_action = get_ingame_action(action)
        observation, reward, done, info = env.step(game_action)

        # Update history
        next_state = preprocess_frame(observation)
        next_state = np.reshape([next_state], (1, 84, 84, 1))
        next_history = np.append(next_state, history[:, :, :, :3], axis=3)
        history = next_history

        reward = np.clip(reward, -1., 1.)
        score += reward
    print("score: ", score)


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", type=str, metavar="MODEL_FILE",
                    help="train the agent to play the game")
parser.add_argument("-p", "--play", type=str, metavar="MODEL_FILE",
                    help="load agent model file to play the game")
args = parser.parse_args()

game = Breakout()
env = gym.make(game.env_name)
if args.train:
    train(env, game, args.train)
elif args.play:
    play(env, game, args.play)
else:
    parser.print_help()
