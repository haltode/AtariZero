import gym
import numpy as np
import skimage.color
import skimage.transform

from dqn_agent import DQNAgent


def preprocess_frame(frame):
    gray_frame = skimage.color.rgb2gray(frame)
    resized_frame = skimage.transform.resize(gray_frame, [84, 84])
    # Store on smallest type possible (uint8) because replay experience will
    # take huge amount of memory, hence using float is very costly
    compact_frame = np.uint8(resized_frame * 255)
    return compact_frame


env = gym.make('BreakoutDeterministic-v4')
agent = DQNAgent()

env.reset()
env.render()

is_done = False
while not is_done:
    frame, reward, is_done, _ = env.step(env.action_space.sample())
    preprocess_frame(frame)
    agent.build_atari_model()
    env.render()
