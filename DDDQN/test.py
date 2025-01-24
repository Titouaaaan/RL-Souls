import gymnasium
import soulsgym
import numpy as np

env = gymnasium.make("SoulsGymIudex-v0")
obs, info = env.reset()
terminated = False

while not terminated:
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(next_obs)

env.close()