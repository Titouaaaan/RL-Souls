'''
SHIT TO FIX:
D:\GAP YEAR\RL-Souls\TorchRL\torchrlenv\Lib\site-packages\torchrl\data\replay_buffers\samplers.py:34: UserWarning: Failed to import torchrl C++ binaries. Some modules (eg, prioritized replay buffers) may not work with your installation. This is likely due to a discrepancy between your package version and the PyTorch version. Make sure both are compatible. Usually, torchrl majors follow the pytorch majors within a few days around the release. For instance, TorchRL 
0.5 requires PyTorch 2.4.0, and TorchRL 0.6 requires PyTorch 2.5.0.
'''

import gymnasium as gym
import numpy as np
import soulsgym
import torch
from torchrl.envs.utils import check_env_specs
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.data import (
    Composite,
    OneHot,
    Bounded,
)
from tensordict import TensorDict
import multiprocessing

# Choose device
is_fork = multiprocessing.get_start_method() == "fork"
device = torch.device("cuda" if torch.cuda.is_available() and not is_fork else "cpu")

# Create base gymnasium env
raw_env = gym.make("SoulsGymIudex-v0")

# Wrap in GymWrapper
wrapped_env = GymWrapper(raw_env)

# Patch to avoid StopIteration: Use GymWrapper's reset() with return_info=True manually
def safe_reset(env):
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    else:
        return result, {}

# Custom observation spec (cleaned for TorchRL ≥ 0.7)
obs_spec = Composite(
    phase=OneHot(n=2, shape=(2,), device=device),
    player_hp=Bounded(shape=(1,), dtype=torch.float32, low=0.1, high=2000.0, device=device),
    player_max_hp=Bounded(shape=(1,), dtype=torch.float32, low=0.0, high=2000.0, device=device),
    player_sp=Bounded(shape=(1,), dtype=torch.float32, low=0.0, high=500.0, device=device),
    player_max_sp=Bounded(shape=(1,), dtype=torch.float32, low=0.0, high=500.0, device=device),
    boss_hp=Bounded(shape=(1,), dtype=torch.float32, low=0.0, high=3000.0, device=device),
    boss_max_hp=Bounded(shape=(1,), dtype=torch.float32, low=0.0, high=3000.0, device=device),
    player_pose=Bounded(shape=(4,), dtype=torch.float32, low=-1000.0, high=1000.0, device=device),
    boss_pose=Bounded(shape=(4,), dtype=torch.float32, low=-1000.0, high=1000.0, device=device),
    camera_pose=Bounded(shape=(6,), dtype=torch.float32, low=-1e6, high=1e6, device=device),
    player_animation=OneHot(n=51, shape=(51,), device=device),
    player_animation_duration=Bounded(shape=(1,), dtype=torch.float32, low=0.0, high=30.0, device=device),
    boss_animation=OneHot(n=33, shape=(33,), device=device),
    boss_animation_duration=Bounded(shape=(1,), dtype=torch.float32, low=0.0, high=30.0, device=device),
    lock_on=OneHot(n=2, shape=(2,), device=device),
)

# Apply to wrapped env
wrapped_env.action_spec = wrapped_env.action_spec.to(device) 
wrapped_env.observation_spec = obs_spec.to(device)

# Create TransformedEnv
env = TransformedEnv(wrapped_env).to(device)

# Print specs
print("Observation spec:", env.observation_spec)
print("Action spec:", env.action_spec)
check_env_specs(env)

n_episodes = 1

for episode in range(n_episodes):
    # Reset safely
    obs, info = safe_reset(env)
    
    terminated = truncated = False
    episode_reward = 0.0

    while not (terminated or truncated):  # Loop until the episode is finished
        # Sample a random action
        action_index = torch.randint(0, env.action_spec.shape[-1], (1,), device=device)
        action = torch.nn.functional.one_hot(action_index, num_classes=env.action_spec.shape[-1]).squeeze(0).to(torch.int64)

        td_input = TensorDict({"action": action.to(device)}, batch_size=[]).to(device=device)

        # Step the environment
        td_output = env.step(td_input)

        # Debug prints
        #print(f"td_output: {td_output}")  # Print the entire output for debugging
        #print(f"\n\n\n\n")
        # Assuming td_input is a TensorDict, you can iterate through its fields
        # Iterate through the fields in the 'next' part of the TensorDict
        """ for key, tensor in td_input['next'].items():
            print(f"{key}: {tensor.cpu().numpy()}") """

        next_tensors = td_input.get('next')

        # Now access reward, terminated, truncated, and done from the 'next' field
        reward = next_tensors.get('reward').item()
        terminated = next_tensors.get('terminated').item()
        truncated = next_tensors.get('truncated').item()
        done = next_tensors.get('done').item()

        episode_reward += reward
        # Print the values
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Done: {done}")

    print(f"Episode {episode + 1} finished with total reward: {episode_reward}")

# Clean up
env.close()
