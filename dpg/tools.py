import numpy as np
import subprocess
import webbrowser
import time
from os import path


def eval_agent(env, agent, obs_dim: tuple, max_iter: int):
    """
    - Evaluation rollout for non-vectorized environment; 1 episode
    :param env: Environment instance
    :param agent: RL agent
    :param act_dim: Action space dimension
    :param obs_dim: Observation space dimension
    :param max_iter: Maximal allowed iterations per episode
    """
    done = False
    observation, _ = env.reset()
    observation = np.expand_dims(observation, axis=0)
    reward_episode = 0
    episode_iter = 0
    while not done:
        action = agent.choose_deterministic_action(observation)

        observation_, reward, done, info, _ = env.step(action)
        reward_episode += reward
        observation_ = np.expand_dims(observation_, axis=0)
        observation = observation_

        if episode_iter > max_iter:
            done = True

        episode_iter += 1

    return reward_episode


def launch_tensorboard(log_dir):
    # Ensure the directory exists
    if not path.exists(log_dir):
        print(f"Error: The directory '{log_dir}' does not exist.")
        return
        # 1. Start TensorBoard as a subprocess
        # --logdir: path to your logs
        # --port: default is 6006, you can change this if needed
    tb_process = subprocess.Popen(
        ['tensorboard', '--logdir', log_dir, '--port', '6006']
    )
    # 2. Give the server a moment to spin up
    print("Starting TensorBoard...")
    time.sleep(4)

    # 3. Open the web browser
    url = "http://localhost:6006/"
    webbrowser.open(url)
    print(f"TensorBoard is running at {url}")
    time.sleep(2)

