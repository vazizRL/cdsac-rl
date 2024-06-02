import gym
import numpy as np
import os
import torch
from dsacv02.agentv02 import Agent as DSACAgent
from sac_implementation.sac_agent_compatible import Agent as SACAgent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tools import smoothing
from environments.linear_env import LinearEnv
from copy import deepcopy


''' Environment constants '''
DEVICE = 'cuda:0'
N_CELLS = 10
CELL_LIST = torch.tensor([[i] for i in range(N_CELLS)], device=DEVICE).unsqueeze(dim=2)
ACTION_LEFT = torch.tensor([-1], device=DEVICE).unsqueeze(dim=1)

''' Agent constants '''
# Action Space for InvertedPendulum-v4
ACTION_DIM = 1
OBSERVATION_DIM = 1
N_KERNELS_ACT = 1
N_KERNELS_CR = 1
# Learning Rates
CR_LR_INI, ACT_LR_INI, ALPHA_LR_INI = 5e-4, 5e-4, 5e-4
CR_LR_FIN, ACT_LR_FIN, ALPHA_LR_FIN = 5e-4, 5e-4, 1e-5
# Standard deviations
EXPONENTIATE = False
CR_MIN_STD, CR_MAX_STD = 0.01, 100.0
ACT_MIN_STD, ACT_MAX_STD = 0.01, 1.0
# Hidden Layers
CR_HL = (32, 32)
ACT_HL = (32, 32)
# Activations
CR_ACTIV = ('relu', 'relu')
ACT_ACTIV = ('relu', 'relu')
# Action boundaries
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
# RL parameters
DOUBLE_Q = False
BATCH_SIZE = 16
T_MAX = N_CELLS*10
TAU = 0.85
STATIC_ALPHA = 0.1                     # 1.0
REWARD_SCALE = 2
GAMMA = 0.99
UPDATE_INTERVAL = 1
AUTO_ALPHA = True
ALPHA_INI = -2.0                    # GIVEN AS x; e**x
MEM_SIZE = 1e4                      # 1e5
N_POL_UPDATE_INTERVAL = 1

'''
Training Parameters
'''
n_tot_steps = 1
MAX_TOTAL_ITER = 9040               # 150000
MAX_EPISODE_ITER = 500
LOG_Y_DIFF_INTERVAL = 3000

''' 
Numerical Parameters
'''
N_SUPP = 31
IBF = 10
SMOOTHING_WEIGHT = 0.85

# Exponentiate hyperparameters if networks output is exponentiated
if EXPONENTIATE:
    e = np.e
    CR_MIN_STD, CR_MAX_STD = e ** CR_MIN_STD, e ** CR_MAX_STD
    ACT_MIN_STD, ACT_MAX_STD = e ** ACT_MIN_STD, e ** ACT_MAX_STD

'''
Saving options
'''
curr_dir = os.getcwd() + '/' + 'analytical_results_test' + '/'
tar_name = 'best_performance.tar'
meta_name = 'agent_meta.txt'
replay_name = 'replay_buffer.pkl'
'''
Instantiate TB
'''
dt = datetime.now()
ts = datetime.timestamp(dt)
event_path = curr_dir + f'/Y-Diff_{ts}'
os.mkdir(event_path)
tb_writer = SummaryWriter(log_dir=event_path, comment='CDSAC_vs_SAC', flush_secs=20)

'''
Calculate true values according to the optimal policy (a=-1 /forall s \in S)
'''
v_pi_optim = [GAMMA**i for i in range(N_CELLS-2)]

if __name__ == '__main__':
    env_dsac = LinearEnv(size=N_CELLS)
    env_sac = LinearEnv(size=N_CELLS)
    diff_dsac_hist = list()
    diff_sac_hist = list()

    agent_dsac = DSACAgent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, n_kernels_act=N_KERNELS_ACT,
                       n_kernels_cr=N_KERNELS_CR, cr_lr_ini=CR_LR_INI, cr_lr_fin=CR_LR_FIN,
                       act_lr_ini=ACT_LR_INI, act_lr_fin=ACT_LR_FIN,
                       alpha_lr_ini=ALPHA_LR_INI, alpha_lr_fin=ALPHA_LR_FIN,
                       value_min_std=CR_MIN_STD, value_max_std=CR_MAX_STD, cr_activ=CR_ACTIV, cr_hl=CR_HL,
                       act_min_std=ACT_MIN_STD, act_max_std=ACT_MAX_STD, act_hl=ACT_HL,
                       act_activ=ACT_ACTIV, action_low=ACTION_LOW, action_up=ACTION_HIGH,
                       batch_size=BATCH_SIZE,
                       t_max=T_MAX, tau=TAU, static_alpha=STATIC_ALPHA, log_alpha_ini=ALPHA_INI,
                       reward_scale=REWARD_SCALE, gamma=GAMMA,
                       update_interval=UPDATE_INTERVAL, auto_alpha=AUTO_ALPHA, double_q=DOUBLE_Q,
                       memory_size=MEM_SIZE, n_supports=N_SUPP, ibf=IBF, device=DEVICE)
    pi_net = deepcopy(agent_dsac.policy)
    q1_net = deepcopy(agent_dsac.q1)
    q2_net = deepcopy(agent_dsac.q1)
    agent_sac = SACAgent(policy_net=pi_net, critic1_net=q1_net, critic2_net=q2_net, actor_lr=ACT_LR_INI,
                         critic_lr=CR_LR_INI, input_dims=OBSERVATION_DIM, gamma=GAMMA,
                         n_actions=ACTION_DIM, max_size=MEM_SIZE, tau=TAU, batch_size=BATCH_SIZE,
                         reward_scale=REWARD_SCALE, auto_temp=AUTO_ALPHA, temp_log_ini=ALPHA_INI, omega=ALPHA_LR_INI,
                         static_temp=STATIC_ALPHA, double_q=DOUBLE_Q)

    curr_best_score_dsac = env_dsac.reward_range[0]
    curr_best_score_sac = env_sac.reward_range[0]

    # Reward smoothing variables
    smooth_reward_iter_n_dsac = 0
    smooth_reward_iter_n_sac = 0
    smooth_reward_last_dsac = 0
    smooth_reward_last_sac = 0
    smoothed_total_dsac = list()
    smoothed_total_sac = list()

    done_dsac = True
    done_sac = True
    reward_epi_dsac = 0
    reward_epi_sac = 0
    for i in range(MAX_TOTAL_ITER):
        # Reset DSAC and SAC if dones are reached respectively
        if done_dsac:
            tb_writer.add_scalar('Reward_DSAC', reward_epi_dsac, n_tot_steps)
            # Calculate smoothed reward and save accordingly
            batch_sm_dsac, smooth_reward_iter_n_dsac, smooth_reward_last_dsac = \
                smoothing(scalars=(reward_epi_dsac,), weight=SMOOTHING_WEIGHT, iter=smooth_reward_iter_n_dsac,
                          last=smooth_reward_last_dsac)
            smoothed_total_dsac.append(batch_sm_dsac)
            smoothed_last_epi_dsac = smoothed_total_dsac[-1][-1]
            if smoothed_last_epi_dsac > curr_best_score_dsac and i > 200:
                curr_best_score_dsac = smoothed_last_epi_dsac
                agent_dsac.save_checkpoint(iter_n=n_tot_steps, path=event_path, tar_name=tar_name, txt_name=meta_name,
                                           replay_txt_name=replay_name)
            print(f'DSAC-Episode {i}, Epi. Reward: {reward_epi_dsac}, Smoothed: {smoothed_last_epi_dsac}')

            reward_epi_dsac = 0
            episode_iter_dsac = 0
            observation_dsac = env_dsac.reset()
            observation_dsac = np.expand_dims(observation_dsac, axis=0)
            done_dsac = False
        if done_sac:
            tb_writer.add_scalar('Reward_SAC', reward_epi_sac, n_tot_steps)
            # Calculate smoothed reward and save accordingly
            batch_sm_sac, smooth_reward_iter_n_sac, smooth_reward_last_sac = \
                smoothing(scalars=(reward_epi_sac,), weight=SMOOTHING_WEIGHT, iter=smooth_reward_iter_n_sac,
                          last=smooth_reward_last_sac)
            smoothed_total_sac.append(batch_sm_sac)
            smoothed_last_epi_sac = smoothed_total_sac[-1][-1]
            if smoothed_last_epi_sac > curr_best_score_sac and i > 200:
                curr_best_score_sac = smoothed_last_epi_sac
                agent_sac.save_models(iter_n=n_tot_steps, path=event_path, tar_name=tar_name, txt_name=meta_name,
                                      replay_txt_name=replay_name)
            print(f'SAC-Episode {i}, Epi. Reward: {reward_epi_sac}, Smoothed: {smoothed_last_epi_sac}')

            reward_epi_sac = 0
            episode_iter_sac = 0
            observation_sac = env_sac.reset()
            observation_sac = np.expand_dims(observation_sac, axis=0)
            done_sac = False

        # Choose actions with DSAC and SAC
        action_dsac, _ = agent_dsac.choose_action(observation_dsac)
        action_sac = agent_sac.choose_action(observation_sac)
        action_dsac = -1 if action_dsac <= 0 else 1
        action_sac = -1 if action_sac <= 0 else 1

        # Get experience with DSAC and SAC and reshape
        observation_dsac_next, reward_dsac, done_dsac, info_dsac, _ = env_dsac.step(action_dsac)
        observation_sac_next, reward_sac, done_sac, info_sac, _ = env_sac.step(action_sac)
        observation_dsac_next = observation_dsac_next.reshape((1, OBSERVATION_DIM))
        observation_sac_next = observation_sac_next.reshape((1, OBSERVATION_DIM))

        # Reformat quantities and save in RB
        reward_dsac = np.asarray(reward_dsac)
        reward_sac = np.asarray(reward_sac)
        done_dsac = np.asarray(done_dsac)
        done_sac = np.asarray(done_sac)
        agent_dsac.save_experience_tupel(observation_dsac, action_dsac, reward_dsac, observation_dsac_next, done_dsac)
        agent_sac.save_experience_tuple(observation_sac, action_sac, reward_sac, observation_sac_next, done_sac)

        # Sum episode Reward
        reward_epi_dsac += reward_dsac
        reward_epi_sac += reward_sac

        # Calculate Overestimation per Interval for all cells
        if n_tot_steps % LOG_Y_DIFF_INTERVAL == 0:
            diff_dsac_chk = list()
            diff_sac_chk = list()
            for cell, true_val in zip(CELL_LIST[1:-1], v_pi_optim):
                if DOUBLE_Q:
                    val_dsac1, _, _ = agent_dsac.q1(cell, ACTION_LEFT)
                    val_dsac2, _, _ = agent_dsac.q2(cell, ACTION_LEFT)
                    val_dsac = 0.5 * (val_dsac1 + val_dsac2)

                    val_sac1, _, _ = agent_sac.q1(cell, ACTION_LEFT)
                    val_sac2, _, _ = agent_dsac.q2(cell, ACTION_LEFT)
                    val_sac = 0.5 * (val_sac1 + val_sac2)
                else:
                    val_dsac = agent_dsac.q1(cell, ACTION_LEFT)
                    val_sac = agent_sac.q1(cell, ACTION_LEFT)

                diff_dsac_i = val_dsac - true_val
                diff_sac_i = val_sac - true_val

                diff_dsac_chk.append(diff_dsac_i.detach().cpu().numpy().squeeze())
                diff_sac_chk.append(diff_sac_i.detach().cpu().numpy().squeeze())
            diff_dsac_hist.append(diff_dsac_chk)
            diff_sac_hist.append(diff_sac_chk)

        tb_info_dsac = agent_dsac.learn(n_learning_iter=N_POL_UPDATE_INTERVAL, step_number=n_tot_steps)
        tb_info_sac = agent_sac.learn()
        for key, value in tb_info_dsac.items():
            tb_writer.add_scalar(key, value, n_tot_steps)
        for key, value in tb_info_sac.items():
            tb_writer.add_scalar(key, value, n_tot_steps)

        observation_dsac = observation_dsac_next
        observation_sac = observation_sac_next

        episode_iter_dsac += 1
        episode_iter_sac += 1
        n_tot_steps += 1

        if n_tot_steps >= MAX_TOTAL_ITER:
            break

    # Convert to np.ndarray and create .npy file names
    print('\n')
    print(' . . . Saving Y-Diffs . . .')
    y_diff_path_dsac = event_path + '/' + 'y_diff_dsac.npy'
    y_diff_path_sac = event_path + '/' + 'y_diff_sac.npy'
    y_diff_dsac_npy = np.asarray(diff_dsac_hist)
    y_diff_sac_npy = np.asarray(diff_sac_hist)
    # Save
    np.save(file=y_diff_path_dsac, arr=y_diff_dsac_npy)
    np.save(file=y_diff_path_sac, arr=y_diff_sac_npy)







