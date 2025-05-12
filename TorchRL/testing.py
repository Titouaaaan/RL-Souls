import gymnasium as gym
import soulsgym
import numpy as np
import torch
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.utils import check_env_specs


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Create a flattened observation space based on an example obs
        obs_sample = self.observation(env.reset()[0])  # first obs from env
        flat_dim = obs_sample.shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def observation(self, obs):
        # Flatten dictionary into a 1D NumPy array
        flat = []
        for v in obs.values():
            arr = np.asarray(v, dtype=np.float32).flatten()
            flat.append(arr)
        return np.concatenate(flat, axis=0)


def make_flattened_env(device):
    # Step 1: Load SoulsGym environment
    raw_env = gym.make("SoulsGymIudex-v0")

    # Step 2: Flatten obs using custom wrapper
    flat_env = FlattenObsWrapper(raw_env)

    # Step 3: Wrap in TorchRL GymWrapper
    torchrl_env = GymWrapper(flat_env).to(device)

    # Step 4: Make it a TransformedEnv (optional — for later transforms)
    transformed_env = TransformedEnv(torchrl_env)

    # Step 5: Check everything works
    check_env_specs(transformed_env)

    print("Observation spec:", transformed_env.observation_spec)
    print("Action spec:", transformed_env.action_spec)

    return transformed_env


    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    env = make_flattened_env(device)

    # Test reset + step
    td = env.reset()
    print(td)
    print("Obs shape after reset:", td["observation"].shape)

    num_episodes = 1
    max_steps = 1000

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep + 1} ===")
        td = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:
            td = env.rand_step(td)
            reward = td["next", "reward"].item()
            done = td["next", "done"].item()
            total_reward += reward
            step += 1

            print(f"Step {step}: reward = {reward}, done = {done}")

        print(f"Total reward: {total_reward} in {step} steps")