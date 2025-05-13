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
from torchrl.modules import ProbabilisticActor, QValueModule, ValueOperator
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule
from torchrl.objectives import DDPGLoss
from torch.optim import Adam
from torchrl.objectives import SoftUpdate, DQNLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
import time

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

    transformed_env = TransformedEnv(torchrl_env).to(device)

    print("Observation spec:", transformed_env.observation_spec)
    print("Action spec:", transformed_env.action_spec)

    return transformed_env

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    env = make_flattened_env(device)
    env.set_seed(0)

    # Test reset + step
    td = env.reset().to(device)
    # print(td)
    # print("Obs shape after reset:", td["observation"].shape)
    num_actions = env.action_spec.shape[0]
    num_obs = env.observation_spec["observation"].shape[0]
    print("Num actions:", num_actions)
    print("Num obs:", num_obs)

    value_mlp = MLP(
        out_features=num_actions,
        num_cells=[64, 64],
    ).to(device)

    value_net = TensorDictModule(
        value_mlp,
        in_keys=["observation"],
        out_keys=["action_value"],
    ).to(device)

    policy = TensorDictSequential(
        value_net,  # writes action values in our tensordict
        QValueModule(spec=env.action_spec)  # Reads the "action_value" entry by default
    ).to(device)

    exploration_module = EGreedyModule(
        env.action_spec, 
        annealing_num_steps=100_000, # start of the decay
        eps_init=0.95 # probability of taking a random action (exploration)
    ).to(device)
    
    policy_explore = TensorDictSequential(
        policy, exploration_module
    ).to(device)

    init_rand_steps = 1e4 # random actions before using the policy (radnom data collection)
    frames_per_batch = 100 # data collection (steps collected per loop)
    optim_steps = 10
    collector = SyncDataCollector(
        env,
        policy_explore,
        frames_per_batch=frames_per_batch, 
        total_frames=-1, # -1 = collect forever
        init_random_frames=init_rand_steps, 
    )
    rb = ReplayBuffer(storage=LazyTensorStorage(100_000))
    
    loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True).to(device)
    optim = Adam(
        loss.parameters(), 
        lr=0.001 # how much to update the weights (low value = slow update but more stable)
    )
    updater = SoftUpdate(
        loss, 
        eps=0.99  # target network update rate (high value means slow update but more stable once again)
    )

    check_env_specs(env) # verify the environment specs is good

    total_count = 0
    total_episodes = 0
    t0 = time.time()

    while total_count < 1e6:
        for i, data in enumerate(collector):
            # print(f'i: {i}')
            # Write data in replay buffer
            rb.extend(data)
            if len(rb) > init_rand_steps:
                # Optim loop (we do several optim steps
                # per batch collected for efficiency)
                for step in range(optim_steps):
                    # print(f'optim step: {step}')
                    sample = rb.sample(128).to(device)
                    loss_vals = loss(sample)
                    loss_vals["loss"].backward()
                    optim.step()
                    optim.zero_grad()
                    # Update exploration factor
                    exploration_module.step(data.numel())
                    # Update target params
                    updater.step()

                    total_count += data.numel()
                    total_episodes += data["next", "done"].sum()
                if i % 10 == 0:
                    mean_reward = data["next", "reward"].mean().item()
                    loss_val = loss_vals["loss"].item()
                    print(f"[Step {i}] RB size: {len(rb)}, Mean reward: {mean_reward}, Loss: {loss_val}, Time: {time.time()-t0:.2f}s")

        t1 = time.time()
    
    print(f'Training stopped after {total_count} steps, {total_episodes} episodes and in {t1-t0}s.')