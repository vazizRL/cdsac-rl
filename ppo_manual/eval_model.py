import gym
import os
import time
from ppo_manual.ppo_low_level import Agent


if __name__ == '__main__':
    curr_dir = os.getcwd()
    do_rendering = True

    if do_rendering:
        env = gym.make('CartPole-v0', render_mode='human')
    else:
        env = gym.make('CartPole-v0')
    horizont = 20                   # Original: 20
    batch_size = 5                  # Or: 5
    n_epochs = 4
    alpha = 0.0005                  # 0.0003
    n_games = 600                   # 600

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        steps = 0
        while not done:
            if steps > 999:
                break
            action, _, _ = agent.choose_action(observation)
            observation_new, reward, done, info, _ = env.step(action)

            score += reward
            observation = observation_new
            steps += 1
        print(f' episode: {i}, reward: {score} \n')


