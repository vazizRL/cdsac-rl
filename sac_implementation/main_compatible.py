"""
- This script is written to test the agent in sac_agent_compatible.py
-
"""
import os
import gym
import numpy as np
from sac_agent_compatible import Agent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dsacv02.actor_critic import Actor, Critic


''' System Initializations '''
device = 'cuda:0'

''' Instantiate TB '''
curr_dir = os.getcwd()
dt = datetime.now()
ts = datetime.timestamp(dt)
event_path = curr_dir + f'/event_{ts}'
os.mkdir(event_path)
tb_writer = SummaryWriter(log_dir=event_path, comment='VanillaDSAC', flush_secs=20)

'''Network Parameters'''
STATE_DIM = 4
ACTION_DIM = 1
N_KERNELS = 1
CR_HL, ACTOR_HL = (256, 256), (256, 256)
CR_ACTIV, ACTOR_ACTIV = ('relu', 'relu'), ('relu', 'relu')

CR_STD_MIN, CR_STD_MAX = 1e-6, 1e-5
ACTOR_STD_MIN, ACTOR_STD_MAX = 1e-6, 1.0
ACTION_MIN, ACTION_MAX = -1.0, 1.0
'''RL Parameters'''
ACTOR_LR = 6e-4
CRITIC_LR = 6e-4
TEMP_LR = 6e-4
GAMMA = 0.95
REPLAY_SIZE = 1e6
TAU = 0.015
BATCH_SIZE = 256
REWARD_SCALE = 2
AUTO_TEMP = False
TEMP_LOG_INI = -2
STATIC_TEMP = 0.3

'''Initiate networks'''
critic1_net = Critic(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_layers=CR_HL, n_kernels=N_KERNELS,
                     activ=CR_ACTIV, value_min_std=CR_STD_MIN, value_max_std=CR_STD_MAX, learnable_weights=False,
                     device=device)
critic2_net = Critic(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_layers=CR_HL, n_kernels=N_KERNELS,
                     activ=CR_ACTIV, value_min_std=CR_STD_MIN, value_max_std=CR_STD_MAX, learnable_weights=False,
                     device=device)
policy_net = Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_layers=ACTOR_HL, n_kernels=N_KERNELS,
                   activation=ACTOR_ACTIV, action_min_std=ACTOR_STD_MIN, action_max_std=ACTOR_STD_MAX,
                   action_low_lim=ACTION_MIN, action_up_lim=ACTION_MAX, learnable_weights=False, device=device)

'''  Saving and Loading Parameters '''
tar_name = 'models.tar'
txt_name = 'agent.txt'
replay_name = 'replay.npy'


if __name__ == '__main__':
    load_checkpoint = False
    render = False
    if load_checkpoint and render:
        # env = gym.make('Pendulum-v1', render_mode='human')
        env = gym.make('CartPole-v1', render_mode='human')

    else:
        # env = gym.make('Pendulum-v1')
        env = gym.make('CartPole-v1')
    agent = Agent(policy_net=policy_net, critic1_net=critic1_net, critic2_net=critic2_net, actor_lr=ACTOR_LR,
                  critic_lr=CRITIC_LR, input_dims=STATE_DIM, gamma=GAMMA, n_actions=ACTION_DIM, max_size=REPLAY_SIZE,
                  tau=TAU, batch_size=BATCH_SIZE,
                  reward_scale=REWARD_SCALE, auto_temp=AUTO_TEMP, temp_log_ini=TEMP_LOG_INI, omega=TEMP_LR,
                  static_temp=STATIC_TEMP
                  )

    iter_tot = 0
    interval_score = 0
    n_games = 5500
    epi_end = 2000
    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models(event_path, tar_name=tar_name, txt_name=txt_name, replay_npy_name=replay_name,
                          load_experience=True)

    for i in range(n_games):
        episode_iter = 0
        observation, _ = env.reset()
        observation = np.expand_dims(observation, axis=0)
        done = False
        score = 0
        interval_score = 0
        while not done:
            action = agent.choose_action(observation)
            action = np.asarray(0, dtype=np.int16) if action <= 0 else np.asarray(1, dtype=np.int16)
            # action = action.astype(np.float64)
            observation_, reward, done, info, _ = env.step(action)
            observation_ = np.expand_dims(observation_, axis=0)
            score += reward
            interval_score += reward
            agent.remember(observation, action, reward, observation_, done)
            iter_tot += 1
            episode_iter += 1
            if iter_tot % 100 == 0:
                print(f'Reward for 100-interval: {interval_score}; with action: {action}')
                interval_score = 0
            if episode_iter > epi_end:
                done = True
            if render:
                env.render()
            if not load_checkpoint:
                tb_info = agent.learn()
                # tb_writer.add_scalar('Reward', reward, iter_tot)
                for key, value in tb_info.items():
                    tb_writer.add_scalar(key, value, iter_tot)
            observation = observation_
        if not load_checkpoint:
            tb_writer.add_scalar('Reward_per_Epsiode', score, i)

        score_history.append(score)
        avg_score = np.mean(score_history[-3:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models(iter_n=iter_tot, path=event_path, tar_name=tar_name, txt_name=txt_name,
                                  replay_txt_name=replay_name)

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)




