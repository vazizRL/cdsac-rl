import gym
import os
import numpy as np
from dqn_agent import Agent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Instantiate tb
curr_dir = os.getcwd()
dt = datetime.now()
ts = datetime.timestamp(dt)
event_path = curr_dir + f'/DQN_event_{ts}'
os.mkdir(event_path)
tb_writer = SummaryWriter(log_dir=event_path, comment='VanillaDQN', flush_secs=20)

# Env. name
env_name = 'LunarLander-v2'
# tar file name
tar_name = 'DQN_Parameters.tar'

# RL hyper-parameters
HL1 = 128
HL2 = 128
GAMMA = 0.99
EPS = 1.0
BATCH_SIZE = 64
EPS_END = 0.01
EPS_DECAY = 5e-5
LR = 1e-4
MEM_SIZE = int(1e5)

if __name__ == '__main__':
    load_checkpoint = False
    render = False
    learn = True
    if load_checkpoint and render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n,
                  fc1_dims=HL1, fc2_dims=HL2, batch_size=BATCH_SIZE, gamma=GAMMA, epsilon=EPS, eps_end=EPS_END,
                  eps_dec=EPS_DECAY, lr=LR, max_mem=MEM_SIZE)

    iter_tot = 0
    interval_score = 0
    n_games = 5500
    epi_end = 1200
    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models(path=curr_dir, tar_name=tar_name, env=env)

    for i in range(n_games):
        episode_iter = 0
        observation, _ = env.reset()
        observation = observation.astype(np.float64)
        done = False
        score = 0
        interval_score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncations, info = env.step(action)
            if episode_iter > epi_end:
                break
            observation_ = observation_.astype(np.float64)
            score += reward
            interval_score += reward
            agent.remember(observation, action, reward, observation_, done)
            iter_tot += 1
            episode_iter += 1
            if iter_tot % 100 == 0:
                print(f'Current total iteration: {iter_tot}')
            if render:
                env.render()
            if learn:
                tb_info = agent.learn()
                tb_writer.add_scalar('Reward', reward, iter_tot)
                for key, value in tb_info.items():
                    tb_writer.add_scalar(key, value, iter_tot)
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-3:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_checkpoint(iter_tot, event_path, tar_name)

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)
