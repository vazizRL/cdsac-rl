import gym
import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dsacv02.tools import smoothing
from elegantrl import *

'''Environment constants'''
gym_env = 'LunarLanderContinuous-v2'
DEVICE = 'cuda:0'

''' Agent constants '''
# Dimensions
ACTION_DIM = 1
OBSERVATION_DIM = 4
# Learning Rates
CR_LR_INI, ACT_LR_INI, ALPHA_LR_INI = 6e-4, 6e-4, 6e-4
CR_LR_FIN, ACT_LR_FIN, ALPHA_LR_FIN = 6e-4, 6e-4, 6e-4
# Standard deviations
ACT_MIN, ACT_MAX = 1e-6, 1.0
# Hidden Layers
CR_ACTIV = ('relu', 'relu')
ACT_ACTIV = ('relu', 'relu')
# Action boundaries
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
# RL parameters
DOUBLE_Q = True
BATCH_SIZE = 256
TAU = 0.015
STATIC_ALPHA = 0.3              # Old 0.2
REWARD_SCALE = 2.0              # Old 0.2
GAMMA = 0.95                    # Old 0.99
TRAIN_INTERVAL = 1
TARGET_UPDATE_INTERVAL = 1
GRADIENT_STEPS = 1
AUTO_ALPHA = True
TARGET_ENTROPY = -0.3
ALPHA_INI = 1                   # Old -1
MEM_SIZE = int(1e6)             # 1e5
SDE = False                     # State-Dependent Exploration
SDE_AT_START = False            # State-Dependent Exploration at start

'''
Training Parameters
'''
N_TOT_STEPS = 0
MAX_TOTAL_ITER = 150000000
N_GAMES = 5500
MAX_EPISODE_ITER = 500
CHK_PROGRESS_INTERVAL = 100

'''
Saving options
'''
curr_dir = os.getcwd() + '/'
tar_name = 'best_performance.tar'
meta_name = 'agent_meta.txt'
replay_name = 'replay_buffer.pkl'

# Instantiate tb
dt = datetime.now()
ts = datetime.timestamp(dt)
event_path = curr_dir + f'/event_{ts}'
os.mkdir(event_path)
tb_writer = SummaryWriter(log_dir=event_path, comment='VanillaDSAC', flush_secs=20)


if __name__ == '__main__':
    env = gym.make(gym_env)

    agent = SAC(policy=MlpPolicy, env=env, gamma=GAMMA, learning_rate=CR_LR_INI, buffer_size=MEM_SIZE,
                learning_starts=BATCH_SIZE+1, train_freq=TRAIN_INTERVAL, target_update_interval=TARGET_UPDATE_INTERVAL,
                batch_size=BATCH_SIZE, tau=TAU, ent_coef='auto', action_noise=None, replay_buffer_class=None,
                replay_buffer_kwargs=None, optimize_memory_usage=False, gradient_steps=GRADIENT_STEPS,
                target_entropy=TARGET_ENTROPY, use_sde=SDE, use_sde_at_warmup=SDE_AT_START, tensorboard_log=None,
                device=DEVICE, _init_setup_model=True
                )

    agent.learn(total_timesteps=100000)

