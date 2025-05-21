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
import gymnasium
import soulsgym
from utils import GameStateTransformer, OneHotEncoder

class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Create a flattened observation space based on an example obs
        obs_sample = self.observation(env.reset()[0])  # first obs from env
        flat_dim = obs_sample.shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        transformer = GameStateTransformer()

    def observation(self, obs):
        # Flatten dictionary into a 1D NumPy array
        #return self.transformer.transform(obs)
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
    transformer = GameStateTransformer()
    env = gym.make("SoulsGymIudex-v0")
    terminated = False
    env.reset()
    while not terminated:
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        new_obs = transformer.transform(next_obs)
        print("Transformed obs:", new_obs, "Len:", len(new_obs))
    env.close()



'''
{
    'phase': 1, 
    'player_hp': array([0.], dtype=float32), 
    'player_max_hp': 454, 1
    'player_sp': array([1.], dtype=float32), 
    'player_max_sp': 95, 
    'boss_hp': array([1004.], dtype=float32), 
    'boss_max_hp': 1037, 
    'player_pose': array([141.30128  , 574.8781   , -68.90828  ,   1.2802318], dtype=float32), 
    'boss_pose': array([140.05641 , 574.2401  , -68.60698 ,  -2.250375], dtype=float32), 
    'camera_pose': array([ 1.4457703e+02,  5.7604156e+02, -6.7674446e+01, -9.3036687e-01, -3.5969186e-01,  7.0962518e-02], dtype=float32), 
    'player_animation': 13, 
    'player_animation_duration': array([0.064], dtype=float32), 
    'boss_animation': 6, 
    'boss_animation_duration': array([1.424], dtype=float32), 
    'lock_on': True
}

player animations=50
boss_animation=30 (or 32?)
'''