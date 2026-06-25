"""
Run toy examples for manual DPG implementation with
"""
from os import getcwd, mkdir
import gym
import numpy as np
from agent_dpg import Agent
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tools import eval_agent, launch_tensorboard


if __name__ == '__main__':
    # Saving Options
    curr_dir = getcwd() + '/Runs/'
    dt = datetime.now()
    ts = datetime.timestamp(dt)
    event_path = curr_dir + f'event_{ts}'
    mkdir(event_path)
    # TensorBoard Writer
    tb_writer = SummaryWriter(log_dir=event_path, comment='ManualDPG', flush_secs=20)
    # Launch Tensorboard Server
    launch_tensorboard(log_dir=event_path)
    # Environment and spaces
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name, render_mode='rgb_array')
    env_eval = gym.make(env_name, render_mode='rgb_array')
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    ACT_MIN = env.action_space.low[0]
    ACT_MAX = env.action_space.high[0]

    # Space Encoding
    TILES_BINS_N = 12       # Old: 12
    TILES_N = 6             # old: 6
    TILES_WIDTH_MULTI = 3   # Old: 3
    TILES_BOX = env.observation_space

    # Networks and Parameters
    # policy_arch = (*obs_shape, 16, *act_shape)
    policy_arch = (*obs_shape, 32, 32, *act_shape)
    # policy_arch = (*obs_shape, 64, *act_shape)
    variance_min = 0.08
    variance = 0.2  # Old: 0.1
    # variance = 1.0  # Old: 0.1
    variance_degeneracy_rate = 1e-5
    # variance_degeneracy_rate = 0

    # Replay Buffer
    REPLAY_SIZE = int(1e5)

    # Training and eval params
    LEARNING_RATE = 3e-4
    # LEARNING_RATE =0.001
    BATCH_SIZE = 32    # old: 32
    GAMMA = 0.99
    DEVICE = 'cuda:0'
    MAX_EPISODE_ITER = 5000
    MAX_ITER = int(1e6)
    CHK_PROGRESS_INTERVAL = 100
    EXPLORATION_BEFORE_LEARNING = 2000
    curr_total_iter = 0
    n_games = 0
    reward_eval_rollout = 0

    # Log Params
    TB_SAVE_INTERVAL = 20
    EVAL_INTERVAL = 1000

    # Instantiate Agent
    agent = Agent(state_dim=obs_shape, act_dim=act_shape, policy_arch=policy_arch,
                  tile_n_bins=TILES_BINS_N, tile_width_multi=TILES_WIDTH_MULTI, tiles_n=TILES_N,
                  tile_box=TILES_BOX, replay_size=REPLAY_SIZE, action_min=ACT_MIN, action_max=ACT_MAX,
                  batch_size=BATCH_SIZE, lr=LEARNING_RATE, gamma=GAMMA, dev=DEVICE)

    # Training loop, capped by MAX_ITER
    while curr_total_iter < MAX_ITER:
        n_games += 1
        episode_iter = 0
        observation, _ = env.reset()
        observation = np.expand_dims(observation, axis=0)
        done = False
        reward_episode = 0
        interval_reward = 0
        # Episode loop
        while not done and curr_total_iter < MAX_ITER:
            curr_total_iter += 1
            episode_iter += 1
            action = agent.choose_action(observation, variance)
            if variance_degeneracy_rate:
                variance -= variance_degeneracy_rate
                variance = max(variance_min, variance)
            observation_next, reward, done, info, _ = env.step(action)
            # observation_next = np.expand_dims(observation_next, axis=0)
            # Check if episode is running too long
            if episode_iter > MAX_EPISODE_ITER:
                done = True
            # Console log info per interval
            if curr_total_iter % CHK_PROGRESS_INTERVAL == 0:
                print(f'Reward for {CHK_PROGRESS_INTERVAL}-interval: {interval_reward}; with action: {action};' + \
                      f'stored transitions: {agent.memory.mem_size}')
                interval_reward = 0
            # Reward quantity management
            interval_reward += reward
            reward_episode += reward
            reward = np.asarray(reward)
            # Convert done to np.ndarray
            done = np.asarray(done, dtype=np.float32)
            # Save experience tuple in replay buffer, each entry is np.ndarray
            agent.remember(state=observation, action=action, reward=reward, state_next=observation_next, done=done)

            # Observation <- Next Observation
            observation = observation_next
            observation = np.expand_dims(observation, axis=0)

            # Train iteration if initial exploration
            if curr_total_iter > EXPLORATION_BEFORE_LEARNING:
                tb_info = agent.learn()
            else:
                tb_info = agent.get_empty_tb_info()

            # TensorBoard log interval
            if curr_total_iter % TB_SAVE_INTERVAL == 0:
                for key, value in tb_info.items():
                    tb_writer.add_scalar(key, value, curr_total_iter)
                tb_writer.add_scalar('Rewards/Reward_Training', reward_episode, curr_total_iter)

            if curr_total_iter % EVAL_INTERVAL == 0:
                reward_eval_rollout = eval_agent(env=env_eval, agent=agent, obs_dim=obs_shape, max_iter=MAX_EPISODE_ITER)
                tb_writer.add_scalar('Rewards/Reward_Eval', reward_eval_rollout, curr_total_iter)

        print('episode', n_games, ', with episode reward %.1f' % reward_episode, ', and last eval rollour: %.1f,'
              %reward_eval_rollout, 'current variance: %.4f' %variance)








