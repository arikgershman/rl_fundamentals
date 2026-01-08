import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vec environment
vec_env = make_vec_env("CarRacing-v3", n_envs=4)

model = PPO("CnnPolicy", vec_env, verbose=1)
    
print("Training PPO on CarRacing")
model.learn(total_timesteps=25000)

# Visualization
obs = vec_env.reset()
try:
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
except KeyboardInterrupt:
    vec_env.close()