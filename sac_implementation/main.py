import pybullet_envs
import gym
import numpy as np
from sac_agent import Agent


if __name__ == '__main__':
    load_checkpoint = True
    if load_checkpoint:
        env = gym.make('InvertedPendulum-v4', render_mode='human')
    else:
        env = gym.make('InvertedPendulum-v4')
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
                  layer1_size=256, layer2_size=256, batch_size=256, tau=0.015, alpha=0.0006, beta=0.0006)
    n_games = 250

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset()
        # observation = observation.astype(np.float32)
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            # observation_ = observation_.astype(np.float32)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if load_checkpoint:
                env.render()
            else:
                agent.learn()
            observation = observation_
            env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-1:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode', i, ', score %.1f' % score, ', avg_score %.1f' % avg_score)




