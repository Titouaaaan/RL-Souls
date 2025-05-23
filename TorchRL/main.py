import time
import os
from tqdm import tqdm
from utils import FlattenObsWrapper
import numpy as np

import gymnasium as gym
import soulsgym
from soulsgym.envs.darksouls3.iudex import IudexEnv
from soulsgym.core.game_state import GameState

import torch
from torch.optim import Adam, AdamW
from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import MLP, QValueModule, EGreedyModule
from torchrl.objectives import SoftUpdate, DQNLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, ListStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

default_checkpoint_dir = "checkpoints"
save_path = "dqn_checkpoint_6.pth"

def save(policy, optim, total_count, default_checkpoint_dir, file_name):
    # Create the full path by joining the default directory and the file name
    full_path = os.path.join(default_checkpoint_dir, file_name)

    # Create the directory if it does not exist
    os.makedirs(default_checkpoint_dir, exist_ok=True)

    # Save the checkpoint
    torch.save({
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "step": total_count
    }, full_path)

    #print(f"Checkpoint saved at {full_path}.")

def compute_custom_reward(game_state: GameState, next_game_state: GameState) -> float:
    """Compute the reward from the current game state and the next game state.

    Args:
        game_state: The game state before the step.
        next_game_state: The game state after the step.

    Returns:
        The reward for the provided game states.
    """
    boss_reward = (game_state.boss_hp - next_game_state.boss_hp) / game_state.boss_max_hp
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

def make_flattened_env(env_name, device, game_speed):
    # Step 1: Load SoulsGym environment
    raw_env = gym.make(env_name, game_speed=game_speed, init_pose_randomization=True) #  device=device, ?

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
    #torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())  # should return True
    print(torch.version.cuda)
    print(torch.cuda.current_device()) # returns 0 (correct)
    print(torch.cuda.get_device_name(torch.cuda.current_device())) # returns cuda:0 if you have one GPU
    print(device)

    env = make_flattened_env(env_name="SoulsGymIudex-v0", device=device, game_speed=3.0)
    #env.set_seed(0)

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
        checkpoint = torch.load(save_path, weights_only=False)

    # observation --> MLP --> Q-values --> QValueModule --> action
    num_cells = [256,256,256]
    value_mlp = MLP(
        out_features=num_actions, # Q values for each action
        num_cells=num_cells, # hidden layers size,
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
    annealing_num_steps=0.3 * training_steps # number of steps to decay the exploration probability
    exploration_module = EGreedyModule(
        env.action_spec, 
        annealing_num_steps=annealing_num_steps, # end of the decay
        eps_init=eps_init, # probability of taking a random action (exploration)\
        eps_end=eps_end
    ).to(device)
    
    policy_explore = TensorDictSequential(
        policy, exploration_module
    ).to(device)

    init_rand_steps = 1e4 # random actions before using the policy (radnom data collection) about 1-5% of RB
    frames_per_batch = 1000 # data collection (steps collected per loop)
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
        storage=LazyTensorStorage( # ListStorage ?
            max_size=size, 
            device=device), 
        sampler=PrioritizedSampler(
            max_capacity=size, 
            alpha=0.8, 
            beta=1.1),
        priority_key="td_error",
        batch_size=200
    )
    
    loss = DQNLoss(
        value_network=policy, 
        loss_function="smooth_l1",
        delay_value=True, # create a target network
        double_dqn=True, # use the target network
        action_space=env.action_spec, 
        
    ).to(device)

    optim = AdamW(
        loss.parameters(), 
        lr=1e-4, # how much to update the weights (low value = slow update but more stable)
        weight_decay=1e-5, # L2 regularization,
        betas=(0.9, 0.999), # momentum
    )
    if LOAD:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])

    updater = SoftUpdate(
        loss, 
        eps=0.995  # target network update rate (high value means slow update but more stable once again)
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

    with tqdm(total=training_steps, desc="Training", unit="steps") as pbar: # initial=collector.total_frames
        for i, data in enumerate(collector):
            # print(f'i: {i}')
            # Write data in replay buffer

            rb.extend(data).to(device)

            pbar.update(data.numel()) # update the progress bar
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
                    
                    rb.update_tensordict_priority(sample)
                
                exploration_module.step(data.numel()) #.to(device) ?
                # Update target params
                updater.step() # .to(device) ?

                total_count += data.numel() # how much data we collected

                pbar.refresh()
                pbar.set_postfix({
                "Reward": data["next", "reward"].mean().item(),
                "Loss": f"{loss_vals['loss'].item():.4f}",
                "Eps": f"{exploration_module.eps}" 
                })
            if i % 100 == 0 and len(rb) > init_rand_steps:
                # Save the model every 50 iterations
                save(policy, optim, total_count, default_checkpoint_dir, save_path)
                    
            if total_count >= training_steps: 
                t1 = time.time()
                break
        t1 = time.time()

        save(policy, optim, total_count, default_checkpoint_dir, save_path)
        print(f"Checkpoint saved at {save_path}.")
        print(f'Training stopped after {total_count} steps in {t1-t0}s.')

def test_agent(policy_path, episodes):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = make_flattened_env(env_name="SoulsGymIudexDemo-v0", device=device, game_speed=1.0)

    num_actions = env.action_spec.shape[0]
    num_obs = env.observation_spec["observation"].shape[0]

    # Define policy (same structure as in training)
    num_cells = [256, 256, 256]
    value_mlp = MLP(
        out_features=num_actions,
        num_cells=num_cells,
        activation_class=torch.nn.ReLU,
    ).to(device)

    value_net = TensorDictModule(
        value_mlp,
        in_keys=["observation"],
        out_keys=["action_value"],
    ).to(device)

    policy = TensorDictSequential(
        value_net,
        QValueModule(spec=env.action_spec),
    ).to(device)

    # Load trained weights
    print(f"Loading policy from {policy_path}")
    checkpoint = torch.load(policy_path, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])

    policy.eval()  # Set to eval mode
    total_rewards = []

    for ep in range(episodes):
        td = env.reset()
        done = False
        terminated = False
        ep_reward = 0.0

        while not (done or terminated):
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                td = policy(td)
                td = env.step(td.clone())

            ep_reward += td["next", "reward"].item()
            done = td["next", "done"].item()
            terminated = td["next", "terminated"].item()

            td = td["next"]  # Advance to the next state

        print(f"Episode {ep+1}: Reward = {ep_reward}")
        total_rewards.append(ep_reward)

    avg_reward = sum(total_rewards) / episodes
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward}")


if __name__ == "__main__":
    file_path = default_checkpoint_dir + "/" + save_path
    #train_agent()
    test_agent(policy_path=file_path, episodes=10)