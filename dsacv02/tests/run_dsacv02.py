import gym
import numpy as np
import os
from dsac_old_versions.dsac_implementation.dsac_agent import Agent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


""" Environment constants"""
# gym_env = 'Pendulum-v1'
gym_env = 'LunarLanderContinuous-v2'

""" Agent constants """
# Action Space for InvertedPendulum-v4
ACTION_DIM = 4
OBSERVATION_DIM = 8
# Learning Rates
CR_LR_INI, ACT_LR_INI, ALPHA_LR_INI = 5e-5, 5e-5, 5e-5
CR_LR_FIN, ACT_LR_FIN, ALPHA_LR_FIN = 6e-5, 1e-6, 1e-6
# Standard deviations
EXPONENTIATE = True
CR_MIN_STD, ACT_MIN_STD = -0.1, -15
CR_MAX_STD, ACT_MAX_STD = 4, 15
# Hidden Layers
CR_HL = (256, 256)
ACT_HL = (256, 256)
# Activations
CR_ACTIV = ('relu', 'relu')
ACT_ACTIV = ('relu', 'relu')
# Action boundaries
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
# RL parameters
BATCH_SIZE = 256
T_MAX = 60000           # Old 20000
TAU = 0.015
STATIC_ALPHA = 1        # Old 0.2
REWARD_SCALE = 2        # Old 0.2
GAMMA = 0.99
UPDATE_INTERVAL = 2
AUTO_ALPHA = True
LOG_ALPHA_INI = 0      # Old 1
MEM_SIZE = 1e5         # 1e5

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
    curr_dir = os.getcwd()
    continue_training = False
    load_checkpoint = False
    render = False
    if load_checkpoint and render:
        agent = Agent(action_dim=ACTION_DIM, obs_dim=OBSERVATION_DIM)
        # agent.load_checkpoint()
        env = gym.make(gym_env, render_mode='human')
        agent = agent.load_checkpoint(path=curr_dir, tar_name=ckpt_name, txt_name=meta_name)
    elif load_checkpoint and not render:
        agent = Agent(action_dim=ACTION_DIM, obs_dim=OBSERVATION_DIM)
        # agent.load_checkpoint()
        env = gym.make(gym_env)
        agent = agent.load_checkpoint(path=curr_dir, tar_name=ckpt_name, txt_name=meta_name)
    else:
        env = gym.make(gym_env)
        agent = Agent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, cr_lr_ini=CR_LR_INI, cr_lr_fin=CR_LR_FIN,
                      act_lr_ini=ACT_LR_INI, act_lr_fin=ACT_LR_FIN, alpha_lr_ini=ALPHA_LR_INI,
                      alpha_lr_fin=ALPHA_LR_FIN, cr_min_log_std=CR_MIN_STD, cr_max_log_std=CR_MAX_STD,
                      cr_activ=CR_ACTIV, cr_hl=CR_HL,
                      act_min_log_std=ACT_MIN_STD, act_max_log_std=ACT_MAX_STD, act_hl=ACT_HL,
                      act_activ=ACT_ACTIV, action_low=ACTION_LOW, action_up=ACTION_HIGH, batch_size=BATCH_SIZE,
                      t_max=T_MAX, tau=TAU, alpha=STATIC_ALPHA, reward_scale=REWARD_SCALE, gamma=GAMMA,
                      update_interval=UPDATE_INTERVAL, auto_alpha=AUTO_ALPHA, log_alpha_ini=LOG_ALPHA_INI,
                      memory_size=MEM_SIZE)

    n_tot_steps = 0
    # n_games = 250
    n_games = 40000
    episode_end = 2000

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        episode_iter = 0
        observation, _ = env.reset()
        done = False
        score = 0
        interval_reward = 0
        while not done:
            action, log_prob_action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            if episode_iter > episode_end:
                done = True
            if n_tot_steps % 100 == 0:
                print(f'Reward for 100-interval: {interval_reward}; with action: {action};' + \
                      f'stores transitions: {agent.memory.mem_cntr}')
                interval_reward = 0
            interval_reward += reward
            score += reward
            reward = np.asarray(reward)
            done = np.asarray(done)
            agent.save_experience_tupel(observation, action, reward, observation_, log_prob_action, done)
            n_tot_steps += 1
            episode_iter += 1
            if render:
                env.render()
            if not load_checkpoint or continue_training:
                tb_info = agent.learn(n_learning_iter=1, step_number=n_tot_steps)
                tb_writer.add_scalar('Reward', reward, n_tot_steps)
                for key, value in tb_info.items():
                    tb_writer.add_scalar(key, value, n_tot_steps)
            observation = observation_
        print(f'@Iter: {n_tot_steps}')
        score_history.append(score)
        avg_score = np.mean(score_history[-3:])

        # if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint or continue_training:
            agent.save_checkpoint(iter_n=i, path=curr_dir, tar_name=ckpt_name, txt_name=meta_name)

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)




