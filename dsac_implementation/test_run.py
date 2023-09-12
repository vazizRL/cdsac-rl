import gym
import os
import numpy as np


curr_dir = os.getcwd()
env = gym.make('CartPole-v0')

# Agent hyperparameters
agent = None
