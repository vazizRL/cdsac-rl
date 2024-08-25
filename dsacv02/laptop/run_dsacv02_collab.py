import gym
import time
import numpy as np
import os
from dsacv02.agentv02 import Agent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dsacv02.tools import smoothing, eval_agent_colab


''' Environment constants '''
SAVE = True
# LOAD_PATH = r"C:\Users\vanya\OneDrive\Desktop\PhD_RL\RL_Framework\dsacv02\event_1724545362.311958".replace('\\', '/')
LOAD_PATH = None
# gym_env = 'Pendulum-v1'
gym_env = 'Walker2d-v4'
DEVICE = 'cuda:0'
DISCRETE = False

''' Agent constants '''
ACTION_DIM = 6
# ACTION_DIM = 1
OBSERVATION_DIM = 17
# OBSERVATION_DIM = 3
N_KERNELS_ACT = 1
N_KERNELS_CR = 1
LEARNABLE_KWEIGHTS = False
# Learning Rates
CR_LR_INI, ACT_LR_INI, ALPHA_LR_INI = 3e-4, 3e-4, 3e-4
CR_LR_FIN, ACT_LR_FIN, ALPHA_LR_FIN = 3e-4, 3e-4, 3e-4      # 6e-4, 6e-4, 6e-4
# Standard deviations
EXPONENTIATE = False
CR_MIN_STD, CR_MAX_STD = 0.01, 100.0           # 0.01, 100.0
ACT_MIN_STD, ACT_MAX_STD = 1e-6, 1.0
# Hidden Layers
CR_HL = (256, 256)
ACT_HL = (256, 256)
# Activations
CR_ACTIV = ('relu', 'relu')     # ('gelu', 'gelu')
ACT_ACTIV = ('relu', 'relu')    # ('gelu', 'gelu')
# Action boundaries
ACTION_LOW = -1.0
# ACTION_LOW = -2.0
ACTION_HIGH = 1.0
# ACTION_HIGH = 2.0
# RL parameters
DOUBLE_Q = True
BATCH_SIZE = 256
T_MAX = 5000                     # Old 20000
TAU = 0.005
STATIC_ALPHA = 1.0              # Old 0.2
REWARD_SCALE = 5.0              # Old 0.2
GAMMA = 0.99                    # Old 0.99
UPDATE_INTERVAL = 1
AUTO_ALPHA = False
ALPHA_INI = 1.0                  # Old 1
MEM_SIZE = 1e6                  # 1e5
N_TRAIN_INTERVAL = 1

'''
Training Parameters
'''
N_TOT_STEPS = 0
MAX_TOTAL_ITER = 1000000
N_GAMES = 5500000000
MAX_EPISODE_ITER = 1000
CHK_PROGRESS_INTERVAL = 100
# WARNING: Below is Experimental
RESET_ENTROPY_ITER = None
EVAL_INTERVAL = 1000
# For downloading event-file in Colab
TB_CHECK_INTERVAL = 250
# For reducing the save interval of TB in training
TB_SAVE_INTERVAL = 20

''' Numerical Parameters'''
N_SUPPORTS = 31                 # 31
IBF = 15

# Exponentiate hyperparameters if networks output is exponentiated
if EXPONENTIATE:
    e = np.e
    CR_MIN_STD, CR_MAX_STD = e ** CR_MIN_STD, e ** CR_MAX_STD
    ACT_MIN_STD, ACT_MAX_STD = e ** ACT_MIN_STD, e ** ACT_MAX_STD

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
event_path_chk = curr_dir + f'/event_chk'
os.mkdir(event_path)
tb_writer = SummaryWriter(log_dir=event_path, comment='C-DSAC', flush_secs=20)
time.sleep(2.1)
tb_writer_chk = SummaryWriter(log_dir=event_path_chk, comment='C-DSAC_chk', flush_secs=20)

if __name__ == '__main__':
    env = gym.make(gym_env)
    env_eval = gym.make(gym_env)
    agent = Agent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, n_kernels_act=N_KERNELS_ACT,
                  n_kernels_cr=N_KERNELS_CR, learnable_kweights=LEARNABLE_KWEIGHTS,
                  cr_lr_ini=CR_LR_INI, cr_lr_fin=CR_LR_FIN,
                  act_lr_ini=ACT_LR_INI, act_lr_fin=ACT_LR_FIN,
                  alpha_lr_ini=ALPHA_LR_INI, alpha_lr_fin=ALPHA_LR_FIN,
                  value_min_std=CR_MIN_STD, value_max_std=CR_MAX_STD, cr_activ=CR_ACTIV, cr_hl=CR_HL,
                  act_min_std=ACT_MIN_STD, act_max_std=ACT_MAX_STD, act_hl=ACT_HL,
                  act_activ=ACT_ACTIV, action_low=ACTION_LOW, action_up=ACTION_HIGH,
                  batch_size=BATCH_SIZE,
                  t_max=T_MAX, tau=TAU, static_alpha=STATIC_ALPHA, log_alpha_ini=ALPHA_INI,
                  reward_scale=REWARD_SCALE, gamma=GAMMA,
                  update_interval=UPDATE_INTERVAL, auto_alpha=AUTO_ALPHA, double_q=DOUBLE_Q,
                  memory_size=MEM_SIZE, n_supports=N_SUPPORTS, ibf=IBF, device=DEVICE)

    if LOAD_PATH:
        N_TOT_STEPS = agent.load_checkpoint(path=LOAD_PATH, tar_name='best_performance.tar', txt_name='agent_meta.txt',
                                            replay_npy_name='replay_buffer.pkl', load_experience=True)

    best_score = env.reward_range[0]
    score_history = []

    # Reward smoothing variables
    smoothing_weight = 0.85
    smooth_reward_last = 0
    smooth_reward_iter_n = 0
    smoothed_total = list()

    for i in range(N_GAMES):
        episode_iter = 0
        observation = env.reset()
        observation = np.expand_dims(observation, axis=0)
        done = False
        reward_episode = 0
        interval_reward = 0
        while not done:
            action, prob_action = agent.choose_action(observation)
            if DISCRETE:
                if ACTION_DIM == 1:
                    action = 0 if action <= 0 else 1
                else:
                    # Single action per time-step for LunarLander-v2
                    action = np.argmax(action)
            else:
                if ACTION_DIM == 1:
                    action = action.squeeze(axis=1)
                else:
                    action = action.squeeze().tolist()
            observation_, reward, done, info = env.step(action)
            observation_ = observation_.reshape((1, OBSERVATION_DIM))
            # observation_ = np.expand_dims(observation_, axis=0)
            if episode_iter > MAX_EPISODE_ITER:
                done = True
                # pass
            if N_TOT_STEPS % CHK_PROGRESS_INTERVAL == 0:
                print(f'Reward for {CHK_PROGRESS_INTERVAL}-interval: {interval_reward}; with action: {action};' + \
                      f'stored transitions: {agent.memory.mem_cntr}')
                interval_reward = 0
            interval_reward += reward
            reward_episode += reward
            reward = np.asarray(reward)
            done = np.asarray(done)
            agent.save_experience_tupel(observation, action, reward, observation_, done)
            N_TOT_STEPS += 1
            episode_iter += 1

            tb_info = agent.learn(n_learning_iter=N_TRAIN_INTERVAL, step_number=N_TOT_STEPS)

            if N_TOT_STEPS % TB_SAVE_INTERVAL == 0:
                for key, value in tb_info.items():
                    tb_writer.add_scalar(key, value, N_TOT_STEPS)
                tb_writer.add_scalar('Rewards/Reward_Training', reward_episode, N_TOT_STEPS)

            if N_TOT_STEPS % EVAL_INTERVAL == 0:
                reward_rollout = eval_agent_colab(env=env_eval, agent=agent, discrete=DISCRETE, act_dim=ACTION_DIM,
                                                  obs_dim=OBSERVATION_DIM, max_iter=MAX_EPISODE_ITER)
                tb_writer.add_scalar('Rewards/Reward_Eval', reward_rollout, N_TOT_STEPS)
            if N_TOT_STEPS % TB_CHECK_INTERVAL == 0:
                print(f'Printing TB_CHK @ episode {i}')
                for key, value in tb_info.items():
                    tb_writer_chk.add_scalar(key, value, N_TOT_STEPS)
                tb_writer_chk.add_scalar('Rewards/Reward_Training', reward_episode, N_TOT_STEPS)

            observation = observation_

        # if N_TOT_STEPS % RESET_ENTROPY_ITER == 0:
        #     agent.reset_entropy(new_log_alpha_val=ALPHA_INI)

        print(f'@Iter: {N_TOT_STEPS}')
        score_history.append(reward_episode)

        batch_sm, smooth_reward_iter_n, smooth_reward_last = \
            smoothing(scalars=(reward_episode,), weight=smoothing_weight, iter=smooth_reward_iter_n,
                      last=smooth_reward_last)
        smoothed_total.append(batch_sm)

        smoothed_last_epi = smoothed_total[-1][-1]
        if smoothed_last_epi > best_score:
            best_score = smoothed_last_epi
            agent.save_checkpoint(iter_n=N_TOT_STEPS, path=event_path, tar_name=tar_name, txt_name=meta_name,
                                  replay_txt_name=replay_name)
        print('episode', i, ', with episode reward %.1f' % reward_episode, ', smoothed total episode reward %.1f'
              % smoothed_last_epi)

        if N_TOT_STEPS >= MAX_TOTAL_ITER:
            break
