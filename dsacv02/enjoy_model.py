import gym
import numpy as np
import os
from dsacv02.agentv02 import Agent
from tools import smoothing

''' Agent'''
ACTION_DIM = 2
OBSERVATION_DIM = 8
N_KERNELS = 1
LEARNABLE_KWEIGHTS = False

''' Environment constants '''
gym_env = 'LunarLander-v2'
DEVICE = 'cuda:0'
DISCRETE = False

'''
Replay Parameters
'''
N_GAMES = 500
N_TOT_STEPS = 0
MAX_TOTAL_ITER = 1500000
# MAX_EPISODE_ITER = 500
MAX_EPISODE_ITER = 500000
CHK_PROGRESS_INTERVAL = 100

'''
Loading Parameters
'''
env_name = 'LunarLander-v2_optim'
event_name = 'MaxHorizont_1000_CrStdMax_1.5_C'
curr_dir = os.getcwd()
loading_path = curr_dir + '/' + 'tests' + '/' + 'DSAC_Runs_Optim' + '/' + env_name + '/' + event_name + '/'
tar_name = 'best_performance.tar'
meta_name = 'agent_meta.txt'
replay_name = 'replay_buffer.pkl'


if __name__ == '__main__':
    render = False
    if render:
        mode = 'human'
    else:
        mode = None
    env = gym.make(gym_env, render_mode=mode, continuous=not DISCRETE)
    # Highly reduced agent instantiation, since all parameters might be replaced
    agent = Agent(obs_dim=OBSERVATION_DIM, action_dim=ACTION_DIM, device=DEVICE, n_kernels_cr=1, n_kernels_act=1,
                  learnable_kweights=LEARNABLE_KWEIGHTS)
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
            action, prob_action = agent.choose_deterministic_action(observation)
            # action, prob_action = agent.choose_action(observation)

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

            observation_, reward, done, info, _ = env.step(action)
            observation_ = observation_.reshape((1, OBSERVATION_DIM))
            # observation_ = np.expand_dims(observation_, axis=0)
            if episode_iter > MAX_EPISODE_ITER:
                done = True
                # pass
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







