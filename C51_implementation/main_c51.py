import gym
import os
import numpy as np
from C51_implementation.c51_agent import Agent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Mode
load_checkpoint = True
render = True
learn = False

# Create new ckpt or specify existing one
curr_dir = os.getcwd()
if learn:
    # Instantiate tb
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    checkpoint_path = curr_dir + '/checkpoint_' + str(ts)
    os.mkdir(checkpoint_path)
    event_path = checkpoint_path + f'/C51_event'
    os.mkdir(event_path)
    tb_writer = SummaryWriter(log_dir=event_path, comment='c51', flush_secs=20)
else:
    checkpoint_path = curr_dir + "/checkpoint_1698086476.527805"


# Env. name
env_name = 'LunarLander-v2'
# File names
tar_name = '/C51_Parameters.tar'
pkl_name = '/agent_params.pkl'

# RL hyper-parameters
N_ATOMS = 51
V_MIN = -25
V_MAX = 25

HL1 = 128
HL2 = 128
GAMMA = 0.99
EPS_START = 1.00        # 1.0
EPS_END = 0.01         # 0.01
DURATION = 6.5e4
BATCH_SIZE = 64
LR = 1e-4
MEM_SIZE = int(6e4)

if __name__ == '__main__':
    if load_checkpoint and render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    agent = Agent(input_dim=env.observation_space.shape, action_dim=env.action_space.n, n_atoms=N_ATOMS,
                  fc1_dim=HL1, fc2_dim=HL2, batch_size=BATCH_SIZE, gamma=GAMMA, start_eps=EPS_START, end_eps=EPS_END,
                  duration=DURATION, lr=LR, max_mem=MEM_SIZE, v_min=V_MIN, v_max=V_MAX)

    iter_tot = 0
    interval_score = 0
    n_games = 5500
    epi_end = 1200
    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_checkpoint(path=checkpoint_path, tar_name=tar_name, pkl_name=pkl_name, greedy=True)

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
                # for key, value in tb_info.items():
                #     tb_writer.add_scalar(key, value, iter_tot)
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-3:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_checkpoint(path=checkpoint_path, tar_name=tar_name, pkl_name=pkl_name)

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)
