import gym
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

# multiprocess environment
env = gym.make('MountainCarContinuous-v0')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1, learning_rate=15e-2, policy_kwargs=dict(act_fun=eval('tf.nn.relu'),
                                                                               net_arch=[50, 50]))
model.learn(total_timesteps=100000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)

    obs, rewards, dones, info = env.step(action)
    print(action)
    env.render()