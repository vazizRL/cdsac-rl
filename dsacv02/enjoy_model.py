import gym
import numpy as np
import os
from dsacv02.agentv02 import Agent
from tools import smoothing

''' Agent'''
ACTION_DIM = 1
OBSERVATION_DIM = 4
N_KERNELS = 1

''' Environment constants '''
# gym_env = 'Pendulum-v1'
gym_env = 'CartPole-v1'
DEVICE = 'cuda:0'

'''
Replay Parameters
'''
N_GAMES = 500
N_TOT_STEPS = 0
MAX_TOTAL_ITER = 15000
MAX_EPISODE_ITER = 500
CHK_PROGRESS_INTERVAL = 100

'''
Loading Parameters
'''
env_name = 'CartPole-v1'
event_name = 'HL_(128,128)'
curr_dir = os.getcwd()
loading_path = curr_dir + '/' + 'tests' + '/' + 'DSAC_Runs' + '/' + env_name + '/' + event_name + '/'
tar_name = 'best_performance.tar'
meta_name = 'agent_meta.txt'
replay_name = 'replay_buffer.pkl'


if __name__ == '__main__':
    render = True
    if render:
        mode = 'human'
    else:
        mode = None
    env = gym.make(gym_env, render_mode=mode)
    # Highly reduced agent instantiation, since all parameters might be replaced
    agent = Agent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, device=DEVICE, n_kernels_cr=1, n_kernels_act=1)
    agent.load_checkpoint(path=loading_path, tar_name=tar_name, txt_name=meta_name, replay_npy_name=replay_name,
                          load_experience=True)

    best_score = env.reward_range[0]
    score_history = []

    # Reward smoothing variables
    smoothing_weight = 0.85
    smooth_reward_last = 0
    smooth_reward_iter_n = 0
    smoothed_total = list()

    for i in range(N_GAMES):
        episode_iter = 0
        observation, _ = env.reset()
        observation = np.expand_dims(observation, axis=0)
        done = False
        reward_episode = 0
        interval_reward = 0
        while not done:
            action, prob_action = agent.choose_action(observation)
            action = 0 if action <= 0 else 1
            observation_, reward, done, info, _ = env.step(action)
            observation_ = observation_.reshape((1, OBSERVATION_DIM))
            # observation_ = np.expand_dims(observation_, axis=0)
            if episode_iter > MAX_EPISODE_ITER:
                # done = True
                pass
            reward_episode += reward
            reward = np.asarray(reward)
            done = np.asarray(done)
            # agent.save_experience_tupel(observation, action, reward, observation_, done)
            N_TOT_STEPS += 1
            episode_iter += 1

            observation = observation_

        print(f'@Iter: {N_TOT_STEPS}')
        score_history.append(reward_episode)

        batch_sm, smooth_reward_iter_n, smooth_reward_last = \
            smoothing(scalars=(reward_episode,), weight=smoothing_weight, iter=smooth_reward_iter_n,
                      last=smooth_reward_last)
        smoothed_total.append(batch_sm)

        smoothed_last_epi = smoothed_total[-1][-1]
        print(f'Last episode reward: {reward_episode}; Smoothed reward last episode: {smoothed_last_epi}')

        if N_TOT_STEPS >= MAX_TOTAL_ITER:
            break







