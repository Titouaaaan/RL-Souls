import gymnasium as gym
import soulsgym
from soulsgym.envs.darksouls3.iudex import IudexEnv
from soulsgym.core.game_state import GameState
import numpy as np
import torch
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.utils import check_env_specs
from tensordict.nn import TensorDictModule
from torchrl.modules import MLP
from torchrl.modules import QValueModule
from tensordict.nn import TensorDictSequential
from torchrl.modules import EGreedyModule
from torch.optim import Adam
from torchrl.objectives import SoftUpdate, DQNLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, ListStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
import time
from tqdm import tqdm
from torchrl.data import TensorDictReplayBuffer

class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Create a flattened observation space based on an example obs
        obs_sample = self.observation(env.reset()[0])  # first obs from env
        flat_dim = obs_sample.shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)

    def observation(self, obs):
        flat = []
        for v in obs.values():
            arr = torch.tensor(v, dtype=torch.float32).flatten()
            flat.append(arr)
        return torch.cat(flat, dim=0)

def save(policy, optim, total_count, file_name="dqn_checkpoint_3sad wa ds qp.pth"):
    torch.save({
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "step": total_count
            }, 
            file_name)
    #print(f"Checkpoint saved at dqn_checkpoint.pth.")

@staticmethod
def compute_custom_reward(game_state: GameState, next_game_state: GameState) -> float:
    """Compute the reward from the current game state and the next game state.

    Args:
        game_state: The game state before the step.
        next_game_state: The game state after the step.

    Returns:
        The reward for the provided game states.
    """
    boss_reward = 2* (game_state.boss_hp - next_game_state.boss_hp) / game_state.boss_max_hp
    player_hp_diff = (next_game_state.player_hp - game_state.player_hp)
    player_reward = player_hp_diff / game_state.player_max_hp
    if next_game_state.boss_hp == 0 or next_game_state.player_hp == 0:
        base_reward = 1 if next_game_state.boss_hp == 0 else -0.1
    else:
        # Experimental: Reward for moving towards the arena center, no reward within 4m distance
        d_center_now = np.linalg.norm(next_game_state.player_pose[:2] - np.array([139., 596.]))
        d_center_prev = np.linalg.norm(game_state.player_pose[:2] - np.array([139., 596.]))
        base_reward = 0.01 * (d_center_prev - d_center_now) * (d_center_now > 4)
    return boss_reward + player_reward + 2 * base_reward

def make_flattened_env(device):
    # Step 1: Load SoulsGym environment
    raw_env = gym.make("SoulsGymIudex-v0", game_speed=3.0) # 

    IudexEnv.compute_reward = staticmethod(compute_custom_reward)

    # Step 2: Flatten obs using custom wrapper
    flat_env = FlattenObsWrapper(raw_env)

    # Step 3: Wrap in TorchRL GymWrapper
    torchrl_env = GymWrapper(flat_env).to(device)

    transformed_env = TransformedEnv(torchrl_env).to(device)

    print("Observation spec:", transformed_env.observation_spec)
    print("Action spec:", transformed_env.action_spec)

    return transformed_env

def train_agent():
    ''' Train the DQN agent on the SoulsGym environment. '''
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())  # should return True
    print(torch.version.cuda)
    print(torch.cuda.current_device()) # returns 0 (correct)
    print(torch.cuda.get_device_name(torch.cuda.current_device())) # returns cuda:0 if you have one GPU
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
    
    training_steps = 2e6 # total training steps

    LOAD = False
    if LOAD:
        print(f'Loading checkpoint (policy + optim + step count)...')
        checkpoint = torch.load("dqn_checkpoint_2.pth", weights_only=False)

    # observation --> MLP --> Q-values --> QValueModule --> action
    value_mlp = MLP(
        out_features=num_actions, # Q values for each action
        num_cells=[512, 512], # hidden layers size,
        activation_class=torch.nn.ReLU, # activation function
    ).to(device)

    value_net = TensorDictModule(
        value_mlp,
        in_keys=["observation"], # obs input
        out_keys=["action_value"], # output Q values stored in "action_value" key
    ).to(device)

    policy = TensorDictSequential(
        value_net,  # computes writes action values in our tensordict
        QValueModule(spec=env.action_spec)  # selects best action (argmax) based on action values
    ).to(device)

    if LOAD:
        policy.load_state_dict(checkpoint["model_state_dict"])

    eps_init=0.995 # probability of taking a random action (exploration)\
    eps_end=0.1
    annealing_num_steps=0.6 * training_steps # number of steps to decay the exploration probability
    exploration_module = EGreedyModule(
        env.action_spec, 
        annealing_num_steps=annealing_num_steps, # end of the decay
        eps_init=eps_init, # probability of taking a random action (exploration)\
        eps_end=eps_end
    ).to(device)
    
    policy_explore = TensorDictSequential(
        policy, exploration_module
    ).to(device)

    init_rand_steps = 1e4 # random actions before using the policy (radnom data collection)
    frames_per_batch = 400 # data collection (steps collected per loop)
    optim_steps = 10 # optim steps per batch collected

    collector = SyncDataCollector(
        env,
        policy_explore,
        frames_per_batch=frames_per_batch, 
        total_frames=-1, # -1 = collect forever
        init_random_frames=init_rand_steps, 
    )

    size = 1_000_000
    rb = TensorDictReplayBuffer(
        storage=ListStorage(size),
        sampler=PrioritizedSampler(max_capacity=size, alpha=0.8, beta=1.1),
        priority_key="td_error",
        batch_size=128
    )
    
    loss = DQNLoss(
        value_network=policy, 
        delay_value=True, # create a target network
        double_dqn=True, # use the target network
        action_space=env.action_spec, 
        
    ).to(device)

    optim = Adam(
        loss.parameters(), 
        lr=3e-4 # how much to update the weights (low value = slow update but more stable)
    )
    if LOAD:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])

    updater = SoftUpdate(
        loss, 
        eps=0.99  # target network update rate (high value means slow update but more stable once again)
    )
    
    if LOAD:
        total_count = checkpoint["step"]
        """ print(total_count)
        print(f"exploration_module._eps device: {exploration_module.eps.device}")
        print(f"device: {device}") """
        epsilon = max(eps_end, eps_init - (eps_init - eps_end) * (total_count / annealing_num_steps)) # this calulcates the epsilon value for our current step
        exploration_module.eps = torch.tensor(epsilon).to(device)
    else:
        total_count = 0
    #total_episodes = 0
    t0 = time.time()

    check_env_specs(env) # verify the environment specs is good

    with tqdm(total=training_steps, desc="Training", unit="steps", initial=total_count) as pbar:
        for i, data in enumerate(collector):
            # print(f'i: {i}')
            # Write data in replay buffer

            rb.extend(data).to(device)
            if len(rb) > init_rand_steps:
                # Optim loop (we do several optim steps
                # per batch collected for efficiency)
                for step in range(optim_steps):
                    
                    # print(f'optim step: {step}')
                    sample = rb.sample().to(device)
                    # print('sample', sample)
                    loss_vals = loss(sample).to(device)
                    loss_vals["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(loss.parameters(), 10)

                    optim.step()
                    optim.zero_grad()
                    # Update exploration factor
                    # print('test', data.numel())
                    exploration_module.step(data.numel()) #.to(device) ?
                    # print('exp', exploration_module)
                    # Update target params
                    updater.step() # .to(device) ?
                    # print('updater', updater)
                    rb.update_tensordict_priority(sample)

                total_count += data.numel() # how much data we collected
                #total_episodes += data["next", "done"].sum()

                pbar.n = total_count
                pbar.refresh()
                pbar.set_postfix({
                "Reward": data["next", "reward"].mean().item(),
                # "Episodes": data["next", "done"].sum().item(), # this is broken bc its resets lol need to find another way
                "Loss": f"{loss_vals['loss'].item():.4f}",
                "Eps": f"{exploration_module.eps}" 
                })
            if i % 100 == 0 and len(rb) > init_rand_steps:
                # Save the model every 50 iterations
                save(policy, optim, total_count)
                    
            t1 = time.time()
            if total_count >= training_steps: 
                break

        save(policy, optim, total_count)
        print(f"Checkpoint saved at dqn_checkpoint.pth.")
        print(f'Training stopped after {total_count} steps in {t1-t0}s.')

def test_agent(policy_path="dqn_checkpoint.pth", episodes=1):
    ''' Test the DQN agent on the SoulsGym environment. '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_flattened_env(device)
    env.set_seed(0)

    # Get dimensions
    num_actions = env.action_spec.shape[0]
    num_obs = env.observation_spec["observation"].shape[0]

    # Build model
    value_mlp = MLP(
        out_features=num_actions,
        num_cells=[64, 128, 256],
    ).to(device)

    value_net = TensorDictModule(
        value_mlp,
        in_keys=["observation"],
        out_keys=["action_value"],
    ).to(device)

    policy = TensorDictSequential(
        value_net,
        QValueModule(spec=env.action_spec)
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(policy_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    # Run test episodes
    for ep in range(episodes):
        td = env.reset().to(device)
        done = False
        ep_reward = 0.0

        while not done:
            with torch.no_grad():
                td = policy(td)
                action = td["action"]
            td = env.step(td)
            reward = td["next", "reward"].item()
            done = td["next", "done"].item()
            ep_reward += reward
            td = td["next"]

        print(f"Episode {ep+1}: reward = {ep_reward:.2f}")


if __name__ == "__main__":
    
    train_agent()
    test_agent(policy_path="dqn_checkpoint.pth", episodes=5)