import gym
import numpy as np
import os
from dsac_implementation.dsac_agent import Agent

""" Agent constants """
# Action Space for InvertedPendulum-v4
ACTION_DIM = 1
OBSERVATION_DIM = 4
# Learning Rates
CR_LR_INI, ACT_LR_INI, ALPHA_LR_INI = 1e-3, 1e-3, 1e-3
CR_LR_FIN, ACT_LR_FIN, ALPHA_LR_FIN = 5e-4, 5e-4, 5e-4
# Standard deviations
CR_MIN_LOG_STD, ACT_MIN_LOG_STD = 0.0, -20.0
CR_MAX_LOG_STD, ACT_MAX_LOG_STD = 2.0, 0.5
# Hidden Layers
CR_HL = (64, 64)
ACT_HL = (64, 64)
# Activations
CR_ACTIV = ('gelu', 'gelu', 'gelu')
ACT_ACTIV = ('gelu', 'gelu', 'gelu')
# Action boundaries
ACTION_LOW = -3
ACTION_HIGH = 3
# RL parameters
BATCH_SIZE = 50
T_MAX = 50
TAU = 0.001
ALPHA = 0.2
REWARD_SCALE = 0.2
GAMMA = 0.99
UPDATE_INTERVAL = 2
AUTO_ALPHA = True
MEM_SIZE = 1e5

if __name__ == '__main__':
    curr_dir = os.getcwd()
    load_checkpoint = False
    if load_checkpoint:
        agent = Agent(action_dim=ACTION_DIM, obs_dim=OBSERVATION_DIM)
        # agent.load_checkpoint()
        env = gym.make('InvertedPendulum-v4', render_mode='human')
    else:
        env = gym.make('InvertedPendulum-v4')
        agent = Agent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, cr_lr_ini=CR_LR_INI, cr_lr_fin=CR_LR_FIN,
                      act_lr_ini=ACT_LR_INI, act_lr_fin=ACT_LR_FIN, alpha_lr_ini=ALPHA_LR_INI,
                      alpha_lr_fin=ALPHA_LR_FIN, cr_min_log_std=CR_MIN_LOG_STD, cr_max_log_std=CR_MAX_LOG_STD,
                      act_min_log_std=ACT_MIN_LOG_STD, act_max_log_std=ACT_MAX_LOG_STD, act_hl=ACT_HL,
                      act_activ=ACT_ACTIV, action_low=ACTION_LOW, action_up=ACTION_HIGH, batch_size=BATCH_SIZE,
                      t_max=T_MAX, tau=TAU, alpha=ALPHA, reward_scale=REWARD_SCALE, gamma=GAMMA,
                      update_interval=UPDATE_INTERVAL, auto_alpha=AUTO_ALPHA, memory_size=MEM_SIZE)

    n_tot_steps = 0
    n_games = 250

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, log_prob_action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            score += reward
            reward = np.asarray(reward)
            done = np.asarray(done)
            agent.save_experience_tupel(observation, action, reward, observation_, log_prob_action, done)

            if load_checkpoint:
                env.render()
            else:
                n_tot_steps += 1
                agent.learn(n_learning_iter=1, step_number=n_tot_steps, clear_mem=False)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-1:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)




