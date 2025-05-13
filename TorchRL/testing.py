import gymnasium as gym
import soulsgym
import numpy as np
import torch
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.utils import check_env_specs
from tensordict.nn import TensorDictModule
from torchrl.modules import Actor
from torchrl.modules import MLP
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Normal
from torchrl.modules import ProbabilisticActor, QValueModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule

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

    num_actions = 20
    num_obs = 26

    value_net = TensorDictModule(
        MLP(out_features=num_actions, num_cells=[32, 32]),
        in_keys=["observation"],
        out_keys=["action_value"],
    ).to(device)

    policy = TensorDictSequential(
        value_net,  # writes action values in our tensordict
        QValueModule(spec=env.action_spec),  # Reads the "action_value" entry by default
    ).to(device)

    policy_explore = TensorDictSequential(policy, EGreedyModule(env.action_spec)).to(device)

    set_exploration_type(ExplorationType.RANDOM)

    rollout = env.rollout(max_steps=100, policy=policy_explore).to(device)
    """ for t in range(rollout.size(0)):
        obs = rollout[t]["observation"]
        print(f"Step {t} observation:", obs) """