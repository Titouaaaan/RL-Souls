import time
import os, shutil
from tqdm import tqdm
from utils import FlattenObsWrapper
import numpy as np
from datetime import datetime

import gymnasium as gym
import soulsgym
from soulsgym.envs.darksouls3.iudex import IudexEnv
from soulsgym.core.game_state import GameState

import torch
from torch.optim import Adam, AdamW
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.utils.tensorboard import SummaryWriter

from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import MLP, QValueModule, EGreedyModule
from torchrl.objectives import SoftUpdate, DQNLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, ListStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

params = {
    "train_env_name": "SoulsGymIudex-v0",  # Environment name
    "test_env_name": "SoulsGymIudexDemo-v0",
    "default_checkpoint_dir": "checkpoints",
    "save_path": "dqn_checkpoint_8.pth",
    "LOAD": True,  # Set to True to load the model
    "training_steps": 3e6,  # Total training steps
    "init_rand_steps": 2e1,  # Random actions before using the policy
    "frames_per_batch": 1000,  # Data collection (steps collected per loop)
    "optim_steps": 30,  # Optim steps per batch collected
    "eps_init": 0.995,  # Probability of taking a random action (exploration)
    "eps_end": 0.1,  # Minimum exploration probability
    "annealing_num_steps_ratio": 0.2,  # Number of steps to decay the exploration probability
    "num_cells": [256, 256, 256],  # Hidden layers size for the MLP
    "size_rb": 1_000_000,  # Size of the replay buffer
    "batch_size": 200,  # Batch size for sampling from the replay buffer
    "lr": 1e-4,  # Learning rate for the optimizer
    "weight_decay": 1e-5,  # L2 regularization
    "betas": (0.9, 0.999),  # Momentum parameters for the optimizer
    "update_rate": 0.995,  # Target network update rate
    "save_checkpoint_every": 50,  # Save the model every N iterations
    "dqn_loss_function": "smooth_l1",  # Loss function for DQN
    "create_target_net": True,  # Create a target network
    "double_dqn": True,  # Use the target network
    "rb_alpha": 0.8,  # Alpha for prioritized replay buffer
    "rb_beta": 1.1,  # Beta for prioritized replay buffer
}

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
        base_reward = 5 if next_game_state.boss_hp == 0 else -0.1
    else:
        # Experimental: Reward for moving towards the arena center, no reward within 4m distance
        d_center_now = np.linalg.norm(next_game_state.player_pose[:2] - np.array([139., 596.]))
        d_center_prev = np.linalg.norm(game_state.player_pose[:2] - np.array([139., 596.]))
        base_reward = 2 * (0.01 * (d_center_prev - d_center_now) * (d_center_now > 4))
    return (1.2 * boss_reward) + player_reward + (base_reward)

def make_flattened_env(env_name, device, game_speed, random_init, phase=None):
    # Step 1: Load SoulsGym environment
    if phase is not None:
        raw_env = gym.make(env_name, game_speed=game_speed, init_pose_randomization=random_init, phase=phase) #  device=device, ?
    else:
        raw_env = gym.make(env_name, game_speed=game_speed, init_pose_randomization=random_init)

    IudexEnv.compute_reward = staticmethod(compute_custom_reward)

    # Step 2: Flatten obs using custom wrapper
    flat_env = FlattenObsWrapper(raw_env)

    # Step 3: Wrap in TorchRL GymWrapper
    torchrl_env = GymWrapper(flat_env).to(device)

    transformed_env = TransformedEnv(torchrl_env).to(device)

    print("Observation spec:", transformed_env.observation_spec)
    print("Action spec:", transformed_env.action_spec)

    return transformed_env

def train_agent(phase, default_checkpoint_dir, save_path):
    ''' Train the DQN agent on the SoulsGym environment. '''
    #torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())  # should return True
    print(torch.version.cuda)
    print(torch.cuda.current_device()) # returns 0 (correct)
    print(torch.cuda.get_device_name(torch.cuda.current_device())) # returns cuda:0 if you have one GPU

    env = make_flattened_env(env_name=params["train_env_name"], device=device, game_speed=3.0, random_init=True, phase=phase)
    #env.set_seed(0)

    # Test reset + step
    _ = env.reset().to(device)
    # print(td)
    # print("Obs shape after reset:", td["observation"].shape)
    num_actions = env.action_spec.shape[0]
    num_obs = env.observation_spec["observation"].shape[0]
    print("Num actions:", num_actions)
    print("Num obs:", num_obs)
    
    training_steps = params['training_steps'] # total training steps

    load_model = params["LOAD"]  # Load the model if True
    if load_model:
        print(f'Loading checkpoint (policy + optim + step count)...')
        to_load = default_checkpoint_dir + "/" + save_path
        checkpoint = torch.load(to_load, weights_only=False)

    # observation --> MLP --> Q-values --> QValueModule --> action
    num_cells = params["num_cells"]  # hidden layers size for the MLP
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

    if load_model:
        policy.load_state_dict(checkpoint["model_state_dict"])

    eps_init = params["eps_init"] # probability of taking a random action (exploration)\
    eps_end = params["eps_end"] # minimum exploration probability
    annealing_num_steps = params["annealing_num_steps_ratio"] *  training_steps # number of steps to decay the exploration probability
    exploration_module = EGreedyModule(
        env.action_spec, 
        annealing_num_steps=annealing_num_steps, # end of the decay
        eps_init=eps_init, # probability of taking a random action (exploration)\
        eps_end=eps_end
    ).to(device)
    
    policy_explore = TensorDictSequential(
        policy, exploration_module
    ).to(device)

    init_rand_steps = params["init_rand_steps"] # random actions before using the policy (radnom data collection) about 1-5% of RB
    frames_per_batch = params["frames_per_batch"] # data collection (steps collected per loop)
    optim_steps = params["optim_steps"] # optim steps per batch collected

    collector = SyncDataCollector(
        env,
        policy_explore,
        frames_per_batch=frames_per_batch, 
        total_frames=-1, # -1 = collect forever
        init_random_frames=init_rand_steps, 
    )

    size = params["size_rb"]  # size of the replay buffer
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage( # ListStorage ?
            max_size=size, 
            device=device), 
        sampler=PrioritizedSampler(
            max_capacity=size, 
            alpha=params["rb_alpha"], # alpha for prioritized replay buffer
            beta=params["rb_beta"]),
        priority_key="td_error",
        batch_size=params["batch_size"], 
    )
    
    loss = DQNLoss(
        value_network=policy, 
        loss_function=params["dqn_loss_function"], # loss function for DQN
        delay_value=params["create_target_net"], # create a target network
        double_dqn=params["double_dqn"], # use the target network
        action_space=env.action_spec, 
        
    ).to(device)

    optim = AdamW(
        loss.parameters(), 
        lr=params["lr"], # how much to update the weights (low value = slow update but more stable)
        weight_decay=params["weight_decay"], 
        betas=params["betas"], # momentum
    )
    if load_model:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])

    updater = SoftUpdate(
        loss, 
        eps=params['update_rate']  # target network update rate (high value means slow update but more stable once again)
    )
    
    if load_model:
        total_count = checkpoint["step"]
        print(f"Resuming training from step {total_count}.")
        """ print(total_count)
        print(f"exploration_module._eps device: {exploration_module.eps.device}")
        print(f"device: {device}") """
        epsilon = max(eps_end, eps_init - (eps_init - eps_end) * (total_count / annealing_num_steps)) # this calulcates the epsilon value for our current step
        exploration_module.eps = torch.tensor(epsilon).to(device)
        print(f"Loaded exploration probability: {exploration_module.eps}")
    else:
        total_count = 0
    #total_episodes = 0
    t0 = time.time()

    check_env_specs(env) # verify the environment specs is good

    # to start the tensorboard: tensorboard --logdir=runs/
    os.makedirs('runs', exist_ok=True)
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writepath = 'runs/{}-{}_'.format('DQN', save_path.split('.')[0]) + timenow
    if os.path.exists(writepath): shutil.rmtree(writepath)
    writer = SummaryWriter(log_dir=writepath)

    with tqdm(total=training_steps, desc="Training", unit="steps") as pbar: # initial=collector.total_frames
        if load_model:
            pbar.update(total_count)
        for i, data in enumerate(collector):
            # print(f'i: {i}')
            # Write data in replay buffer

            rb.extend(data).to(device)

            pbar.update(data.numel()) # update the progress bar
            if len(rb) > init_rand_steps or load_model:
                # Optim loop (we do several optim steps
                # per batch collected for efficiency)
                avg_loss = torch.empty(0).to(device)
                avg_reward = torch.empty(0).to(device)
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

                    # for tensorboard logging
                    avg_loss = torch.cat([avg_loss, loss_vals["loss"].detach().unsqueeze(0)])
                    avg_reward = torch.cat([avg_reward, data["next", "reward"].detach().unsqueeze(0)])

                
                avg_loss = torch.mean(avg_loss)
                avg_reward = torch.mean(avg_reward)

                writer.add_scalar('Loss/train', avg_loss, total_count)
                writer.add_scalar('Reward/train', avg_reward, total_count)

                exploration_module.step(data.numel()) #.to(device) ?
                # Update target params
                updater.step() # .to(device) ?

                total_count += data.numel() # how much data we collected

                pbar.refresh()
                pbar.set_postfix({
                "Reward": f"{avg_reward:.4f}",
                "Loss": f"{avg_loss:.4f}",
                "Eps": f"{exploration_module.eps}" 
                })
            if i % params["save_checkpoint_every"] == 0 and len(rb) > init_rand_steps:
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
    env = make_flattened_env(env_name=params["test_env_name"], device=device, game_speed=1.0, random_init=False)

    num_actions = env.action_spec.shape[0]
    num_obs = env.observation_spec["observation"].shape[0]

    # Define policy (same structure as in training)
    num_cells = params["num_cells"]  # hidden layers size for the MLP
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
    file_path = params["default_checkpoint_dir"] + "/" + params["save_path"]
    #train_agent(phase=1, default_checkpoint_dir=default_checkpoint_dir, save_path=save_path)
    #train_agent(phase=2, default_checkpoint_dir=params["default_checkpoint_dir"], save_path=params["save_path"])
    test_agent(policy_path=file_path, episodes=10)