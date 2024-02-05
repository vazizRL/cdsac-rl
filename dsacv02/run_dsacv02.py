import gym
import numpy as np
import os
from dsacv02.agentv02 import Agent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


""" Environment constants"""
gym_env = 'Pendulum-v1'
DEVICE = 'cuda:0'

""" Agent constants """
# Action Space for InvertedPendulum-v4
ACTION_DIM = 1
OBSERVATION_DIM = 3
N_KERNELS = 1
# Learning Rates
CR_LR_INI, ACT_LR_INI, ALPHA_LR_INI = 1e-4, 1e-4, 1e-4
CR_LR_FIN, ACT_LR_FIN, ALPHA_LR_FIN = 1e-5, 1e-5, 1e-5
# Standard deviations
EXPONENTIATE = False
CR_MIN_STD, CR_MAX_STD = 0.1, 10.0
ACT_MIN_STD, ACT_MAX_STD = 0.01, 3
# Hidden Layers
CR_HL = (64, 64)
ACT_HL = (64, 64)
# Activations
CR_ACTIV = ('gelu', 'gelu')
ACT_ACTIV = ('gelu', 'gelu')
# Action boundaries
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
# RL parameters
BATCH_SIZE = 64
T_MAX = 15000           # Old 20000
TAU = 1.0
STATIC_ALPHA = 1        # Old 0.2
REWARD_SCALE = 2        # Old 0.2
GAMMA = 0.99
UPDATE_INTERVAL = 1
AUTO_ALPHA = True
ALPHA_INI = 2.71      # Old 1
MEM_SIZE = 1e5         # 1e5
N_POL_UPDATE_INTERVAL = 1

# Exponentiate hyperparameters if networks output is exponentiated
if EXPONENTIATE:
    e = np.e
    CR_MIN_STD, CR_MAX_STD = e ** CR_MIN_STD, e ** CR_MAX_STD
    ACT_MIN_STD, ACT_MAX_STD = e ** ACT_MIN_STD, e ** ACT_MAX_STD

# Saving options
curr_dir = os.getcwd() + '/'
ckpt_name = 'best_performance.tar'
meta_name = 'agent_meta.txt'

# Instantiate tb
dt = datetime.now()
ts = datetime.timestamp(dt)
event_path = curr_dir + f'/event_{ts}'
os.mkdir(event_path)
tb_writer = SummaryWriter(log_dir=event_path, comment='VanillaDSAC', flush_secs=20)

if __name__ == '__main__':
    env = gym.make(gym_env)
    agent = Agent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, n_kernels=N_KERNELS, cr_lr_ini=CR_LR_INI,
                  cr_lr_fin=CR_LR_FIN, act_lr_ini=ACT_LR_INI, act_lr_fin=ACT_LR_FIN, alpha_lr_ini=ALPHA_LR_INI,
                  alpha_lr_fin=ALPHA_LR_FIN, value_min_std=CR_MIN_STD, value_max_std=CR_MAX_STD,
                  cr_activ=CR_ACTIV, cr_hl=CR_HL,
                  act_min_std=ACT_MIN_STD, act_max_std=ACT_MAX_STD, act_hl=ACT_HL,
                  act_activ=ACT_ACTIV, action_low=ACTION_LOW, action_up=ACTION_HIGH, batch_size=BATCH_SIZE,
                  t_max=T_MAX, tau=TAU, alpha=STATIC_ALPHA, reward_scale=REWARD_SCALE, gamma=GAMMA,
                  update_interval=UPDATE_INTERVAL, auto_alpha=AUTO_ALPHA, log_alpha_ini=ALPHA_INI,
                  memory_size=MEM_SIZE, device=DEVICE)

    n_tot_steps = 0
    n_games = 250
    episode_end = 2000
    backup_info_interval = 100

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        episode_iter = 0
        observation, _ = env.reset()
        observation = np.expand_dims(observation, axis=0)
        done = False
        score = 0
        interval_reward = 0
        while not done:
            action, prob_action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            observation_ = observation_.reshape((1, OBSERVATION_DIM))
            # observation_ = np.expand_dims(observation_, axis=0)
            if episode_iter > episode_end:
                break
            if n_tot_steps % backup_info_interval == 0:
                print(f'Reward for {backup_info_interval}-interval: {interval_reward}; with action: {action};' + \
                      f'stored transitions: {agent.memory.mem_cntr}')
                interval_reward = 0
            interval_reward += reward
            score += reward
            reward = np.asarray(reward)
            done = np.asarray(done)
            agent.save_experience_tupel(observation, action, reward, observation_, done)
            n_tot_steps += 1
            episode_iter += 1

            tb_info = agent.learn(n_learning_iter=N_POL_UPDATE_INTERVAL, step_number=n_tot_steps)
            tb_writer.add_scalar('Reward', reward, n_tot_steps)
            for key, value in tb_info.items():
                tb_writer.add_scalar(key, value, n_tot_steps)
            observation = observation_
        print(f'@Iter: {n_tot_steps}')
        score_history.append(score)
        avg_score = np.mean(score_history[-3:])

        # if avg_score > best_score:
        best_score = avg_score

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)




