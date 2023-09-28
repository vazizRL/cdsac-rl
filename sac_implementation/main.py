# import pybullet_envs
import gym
import numpy as np
from sac_agent import Agent


if __name__ == '__main__':
    load_checkpoint = True
    render = True
    if load_checkpoint and render:
        env = gym.make('Pendulum-v1', render_mode='human')
    else:
        env = gym.make('Pendulum-v1')
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
                  layer1_size=256, layer2_size=256, batch_size=256, tau=0.015, alpha=0.0006, beta=0.0006)

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
            action = action.astype(np.float64)
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
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-3:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)




