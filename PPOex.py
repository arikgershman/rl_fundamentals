import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("CarRacing-v3", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)

obs = vec_env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    for d in dones:
        done = done or d
    vec_env.render("human")
