import sys
import gym
from stable_baselines import PPO2
from D3_Integrated import PandaRobotEnv_
from stable_baselines.common.vec_env import DummyVecEnv
import time

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]  # Provide the path to your custom model
    else:
        path = 'C:/Users/XMG/Desktop/Master_+_Bildung/Masterarbeit_DONE/IK_RL/Results/PPO2/D3_III/checkpoint_4700.zip'

    if len(sys.argv) > 2:
        iterations = int(sys.argv[2])
    else:
        iterations = 7000

    # Todo: following line could be adjusted to be dependent on parameters used during training.
    env = PandaRobotEnv_(renders=True, fixedActionRepetitions=True, evalFlag=True)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run, hence vectorize

    try:
        model = PPO2.load(path)
    except ValueError:
        print(
            '\nError: Make sure to have the pre-trained models available or to provide a valid path to a custom model as an argument.\n')
        sys.exit()

    # Enjoy trained agent
    obs = env.reset()
    time_step_counter = 0
    while time_step_counter < iterations:
        env.envs[0].set_step_counter(time_step_counter)
        action, _ = model.predict(obs)
        obs, _, _, info = env.step(action)  # Assumption: eval conducted on single env only!

        # reward, time_step_counter, done = info[0][:]
        time.sleep(0.1)
        # if done:
        #     obs = env.reset()
