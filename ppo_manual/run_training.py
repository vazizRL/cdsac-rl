import gym
import os
import numpy as np
from ppo_manual.ppo_low_level import Agent

if __name__ == '__main__':
    curr_dir = os.getcwd()

    env = gym.make('CartPole-v0')
    # env = gym.make('InvertedPendulum-v4')
    horizont = 20                   # Original: 20
    batch_size = 5                  # Or: 5
    n_epochs = 5
    alpha = 0.0005                  # 0.0005
    n_games = 300                   # 600

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)

    figure_file = curr_dir + '/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            if learn_iters > 2000:
                break
            action, prob, val = agent.choose_action(observation)
            observation_new, reward, done, info, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % horizont == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_new

        score_history.append(score)
        avg_score = np.mean(score_history[-1:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(f'episode {i}: score {score}, avg_score {avg_score}, time_steps {n_steps} lr_steps {learn_iters} \n')
















