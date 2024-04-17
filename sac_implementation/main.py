import os
import gym
import numpy as np
from sac_agent import Agent
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# Instantiate tb
curr_dir = os.getcwd()
dt = datetime.now()
ts = datetime.timestamp(dt)
event_path = curr_dir + f'/event_{ts}'
os.mkdir(event_path)
tb_writer = SummaryWriter(log_dir=event_path, comment='VanillaDSAC', flush_secs=20)

if __name__ == '__main__':
    load_checkpoint = False
    render = False
    if load_checkpoint and render:
        # env = gym.make('Pendulum-v1', render_mode='human')
        env = gym.make('CartPole-v1', render_mode='human')

    else:
        # env = gym.make('Pendulum-v1')
        env = gym.make('CartPole-v1')
    # agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
    #               layer1_size=256, layer2_size=256, batch_size=256, tau=0.015, alpha=0.0006, beta=0.0006)
    agent = Agent(input_dims=(4,), env=env, n_actions=1,
                  layer1_size=256, layer2_size=256, batch_size=256, tau=0.015, alpha=0.0006, beta=0.0006,
                  auto_temp=True, ini_temp=0.08, omega=0.0006, static_temp=1,
                  )

    iter_tot = 0
    interval_score = 0
    n_games = 5500
    epi_end = 2000
    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):

        episode_iter = 0
        observation, _ = env.reset()
        observation = observation.astype(np.float64)
        done = False
        score = 0
        interval_score = 0
        while not done:
            action = agent.choose_action(observation)
            action = np.asarray(0, dtype=np.int16) if action <= 0 else np.asarray(1, dtype=np.int16)
            # action = action.astype(np.float64)
            observation_, reward, done, info, _ = env.step(action)
            observation_ = observation_.astype(np.float64)
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
                agent.save_models()

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)




