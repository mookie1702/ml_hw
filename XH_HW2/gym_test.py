## Check Gym environment

import gymnasium as gym
import numpy as np

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
# env = gym.make("ALE/SpaceInvaders-ram-v5")
observation, info = env.reset(seed=15)

# action_dim = env.action_space.n
# state_len = len(env.observation_space.shape)
# state_dim = 1
# for i in range(state_len):
#     state_dim = state_dim * env.observation_space.shape[i]

# print(f"action dim: {action_dim}, state dim: {state_dim}")
for step in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(f'step: {step}', end="\r")

    if terminated or truncated:
        observation, info = env.reset()

env.close()