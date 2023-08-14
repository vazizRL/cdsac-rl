import os, time
import json, csv
import gym
import tensorflow as tf
from stable_baselines import PPO2
from S30_HP import PandaRobotEnv_
from callback import callback
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from datetime import datetime
from stable_baselines.common.vec_env import SubprocVecEnv
import argparse
import numpy as np
import random
import string
from typing import Dict, Any

def random_string(length=10):
    """
        Generate a random string of given length
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


now = datetime.now()

# Specify save directories
ALGO = "PPO2"
ENV_NAME = "PandaController"
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
MODEL_ID = ENV_NAME + TIME_STAMP + random_string()
PATH = "Results/" + ALGO + "/" + MODEL_ID + "/"
SUFFIX = "final_model"
SAVE_MODEL_DESTINATION = PATH + SUFFIX          # For saving checkpoints and final model
TENSORBOARD_LOCATION = PATH + "tensorboard/"    # For tensorboard usage

# Specify load directories
loading = False
PATH_TO_MODEL = 'C:/Users/vaziz/PycharmProjects/Rechenknecht_III/Results/PPO2/PandaController_2021_17_02__16_47_11__396785ncmankphgq/checkpoint_2100.zip'
PATH_TO_TB_LOGFILE = 'C:/Users/vaziz/PycharmProjects/Rechenknecht_III/Results/PPO2/PandaController_2021_17_02__16_47_11__396785ncmankphgq/tensorboard/PPO2_1'


# DEFAULTS:

RENDER = False
FIXED_NUM_REPETITIONS = True
CHECKPOINT_FREQUENCY = 100  # old: 100; dependent on number of agent actions
DIST_SPECIFICATION = [0, 'A']
RENDER = False
POLICY = 'MlpPolicy'
ACT_FUN = 'tf.nn.tanh'
VERBOSE = 1
TOTAL_TIMESTEPS = 1000

net_arch = [150, 150]
learning_rate = 2.5e-4
horizon = 40
nmini = 8
clipr = 0.2
gamma = 0.95
lambda_ = 0.95
log_train_frequency = 5
log_train_progress_data = ['Update_nr',  # Current count of weight updates performed so far
                           'Grasps',  # Nr of grasps over last X weight updates
                           'Avg_grasp_time_steps',  # Average of time steps needed in a simulation to get from init
                           # pose to attaining goal, averaged over the time steps recorded
                           # for all successful grasps over last X weight updates
                           'Std_grasp_time_steps',
                           'Max_grasp_time_steps',
                           'Min_grasp_time_steps',
                           'Total_time_steps'  # Total time steps simulated so far
                           ]


def create_dir(path=PATH):
    """
        Ensure that a given path exists.
        :param direct: Directory to be created when necessary.
        :return: -
    """
    if not os.path.exists(path):
        os.makedirs(path)


def setup_train_log_file(path=PATH):
    create_dir(path)
    with open(path+"training_eval.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel', quoting=csv.QUOTE_ALL)
        wr.writerow(log_train_progress_data)

# Create a vectorized environment compatible with SB implementation
def make_custom_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = PandaRobotEnv_(renders=RENDER,
                    fixedActionRepetitions=FIXED_NUM_REPETITIONS,
                    distSpecifications=DIST_SPECIFICATION)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init

if __name__ == '__main__':

    n_cpu = 4

    params = dict(
        tensorboard_log=TENSORBOARD_LOCATION
    )
    # Define Search Space and Create Search Grid
    learning_rate = [0.0001, 0.0005, 0.001]
    net_arch = [150, 300]
    cliprange = [0.1, 0.2]
    gamma = [0.8, 0.9]
    lambda_ = [0.75, 0.9]
    horizon = [20, 30, 50, 70]
    nmini = [5, 10]
    # learning_rate = [0.0001, 0.00025, 0.0005, 0.00075, 0.001]
    # net_arch = [50, 150, 200, 300]
    # cliprange = [0.1, 0.125, 1.5, 1.75, 0.2]
    # gamma = [0.25, 0.5, 0.75, 0.9]
    # lambda_ = [0.25, 0.5, 0.75, 0.9]
    # horizon = [20, 30, 50, 70]
    # nmini = [5, 10, 15, 20, 25]

    c1 = np.array([learning_rate])
    c2 = np.array([net_arch])
    c3 = np.array([cliprange])
    c4 = np.array([gamma])
    c5 = np.array([lambda_])
    c6 = np.array([horizon])
    c7 = np.array([nmini])

    hp_matrix = np.array(np.meshgrid(c1, c2,c3,c4,c5,c6,c7)).T.reshape(-1, 7)
    print(hp_matrix.shape)
    create_dir(TENSORBOARD_LOCATION)
    history_score = []
    for i in range(len(hp_matrix)):
        # create_dir(TENSORBOARD_LOCATION + '__' + str(i))
        setup_train_log_file()

        lr, net_arch, clipr, gamma, lambda_, horizon, nmini = hp_matrix[i]
        net_arch = int(net_arch)
        #
        # env = PandaRobotEnv_(renders=RENDER,
        #                     fixedActionRepetitions=FIXED_NUM_REPETITIONS,
        #                     distSpecifications=DIST_SPECIFICATION)

        env = SubprocVecEnv([make_custom_env(i) for i in range(n_cpu)])

        # env = DummyVecEnv([lambda: env])   # The algorithms require a vectorized environment to run, hence vectorize

        # Check wor whether to continue training of a previously created & trained model

            # Create new PPO agent
            # Adjust horizon to the number of cores used.
        model = PPO2(policy=POLICY,
                     env=env,
                     n_steps=40,
                     nminibatches=nmini,
                     cliprange_vf=None,
                     ent_coef=0.01,
                     vf_coef=0.5,
                     max_grad_norm=0.5,
                     lam=lambda_,
                     gamma=gamma,
                     noptepochs=4,
                     cliprange=clipr,
                     policy_kwargs=dict(act_fun=eval(ACT_FUN),
                                        net_arch=[net_arch, net_arch]),
                     verbose=VERBOSE,
                     learning_rate=lr,
                     tensorboard_log=TENSORBOARD_LOCATION)

        model.path = PATH
        model.checkpoint_frequency = CHECKPOINT_FREQUENCY
        model.log_train_progress_frequency = log_train_frequency

        # Retrieve the environment
        env = model.get_env()

        # Train the agent

        model.learn(total_timesteps=int(TOTAL_TIMESTEPS), callback=callback)

        score = env.retrieve_average_reward()
        history_score.append(score)
        # Save the agent
        create_dir()
        model.save(SAVE_MODEL_DESTINATION + '_' + str(i))

    best_run = np.argmax(history_score)
    print('Highest reward in Iteration {} \n'.format(best_run))
    print('With the parameters: {}'.format(hp_matrix[best_run]))

