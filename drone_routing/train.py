import os
from stable_baselines3 import PPO
from drone_routing import DroneRoutingEnv, DroneEnvConfig
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

config = DroneEnvConfig.from_yaml(os.path.join(ROOT_DIR, "configs/default_config.yaml"))
print(config.model_dump())
env = DroneRoutingEnv.from_config(config)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

obs, _ = env.reset()
for _ in range(200):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        print("######### Terminated ###########")
        env.print_results()
        if np.all(env.current_demands <= 0):
            break
        obs = env.reset()