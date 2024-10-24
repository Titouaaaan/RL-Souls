import gymnasium
import soulsgym

import math
import random
import numpy
import csv
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from soulsgym.core.speedhack import inject_dll, SpeedHackConnector

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

env = gymnasium.make("SoulsGymIudex-v0")

'''dll_path = Path("D:\GAP YEAR\RL-Souls\soulsenv\Lib\site-packages\soulsgym\core\speedhack\_C\SpeedHackDLL.dll")

inject_dll("DarkSoulsIII.exe", dll_path)

# Initialize the speedhack connector
speedhack = SpeedHackConnector(process_name="DarkSoulsIII.exe")

# Set the game speed (e.g., 2.0 for double speed)
speedhack.set_game_speed(2.0)'''

def preprocess_state(state_dict):
    state_values = []
    # for debugging atm
    
    # Add phase, player_hp, player_max_hp, player_sp, player_max_sp, boss_hp, boss_max_hp
    state_values.append(state_dict['phase'])
    state_values.extend(state_dict['player_hp'].tolist())  # Convert arrays to lists
    state_values.append(state_dict['player_max_hp'])
    state_values.extend(state_dict['player_sp'].tolist())
    state_values.append(state_dict['player_max_sp'])
    state_values.extend(state_dict['boss_hp'].tolist())
    state_values.append(state_dict['boss_max_hp'])
    
    # Add player_pose, boss_pose, camera_pose
    state_values.extend(state_dict['player_pose'].tolist())
    state_values.extend(state_dict['boss_pose'].tolist())
    state_values.extend(state_dict['camera_pose'].tolist())
    
    # Add player_animation, player_animation_duration, boss_animation, boss_animation_duration
    state_values.append(state_dict['player_animation'])
    state_values.extend(state_dict['player_animation_duration'].tolist())
    state_values.append(state_dict['boss_animation'])
    state_values.extend(state_dict['boss_animation_duration'].tolist())
    
    # Convert lock_on to 0 or 1 (boolean to int)
    state_values.append(int(state_dict['lock_on']))

    # Convert the list to a tensor
    state_tensor = torch.tensor(state_values, dtype=torch.float32, device=device)
    
    return state_tensor.unsqueeze(0).flatten()  # Add batch dimension

# Function to log data into a CSV file
def log_episode_data(episode, steps, avg_reward, avg_loss, filename="training_log.csv"):
    # Write header only once (for the first episode)
    if episode == 0:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Steps", "Avg Reward", "Avg Loss"])
    
    # Append data for each episode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, steps, avg_reward, avg_loss])

episode_losses = []
episode_rewards = []

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 5000
TAU = 0.01
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
state = preprocess_state(state)
n_observations = len(state)
print('Amount of obs: ', n_observations)
print(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

LOAD_PROLICY = True

if LOAD_PROLICY:
    policy_net.load_state_dict(torch.load("dqn_iudex_policy.pth"))

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # change vale here to lower for more stability?
    optimizer.step()

    return loss

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 5000
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = preprocess_state(state)
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    state = state.clone().detach().unsqueeze(0)
    # state

    total_reward = 0
    total_loss = 0
    episode_steps = 0
    if i_episode%500==0:
        torch.save(policy_net.state_dict(), "dqn_iudex_policy.pth")

    for t in count():
        action = select_action(state)

        #maybe add more intermediate rewards here?
        observation, reward, terminated, truncated, _ = env.step(action.item())

        observation = preprocess_state(observation)
        reward = torch.tensor([reward], device=device)
        # print(f"reward at time {t}: {reward}")

        total_reward += reward.item()
        episode_steps += 1

        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        loss = optimize_model()
    
        if loss is not None:
            total_loss += loss.item()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:

            # Calculate average reward and loss for the episode
            avg_reward = total_reward / episode_steps
            avg_loss = total_loss / episode_steps if episode_steps > 0 else 0
            print(f'Episode {i_episode}: episode steps={episode_steps}, avg_reward={avg_reward}, avg_loss={avg_loss}')

            # Save the episode data to the log file
            log_episode_data(i_episode, episode_steps, avg_reward, avg_loss)

            episode_rewards.append(avg_reward)
            episode_losses.append(avg_loss)

            episode_durations.append(t + 1)
            break

print('Complete')

# Save the trained policy network
torch.save(policy_net.state_dict(), "dqn_iudex_policy.pth")

'''
# POTENTIAL WAY OF EVALUATING THE MODEL - TO DO LATER
# Load the saved policy network
policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("dqn_iudex_policy.pth"))

# Set the network to evaluation mode
policy_net.eval()

# Evaluate the agent (you can use a loop or a single run)
state, info = env.reset()
state = preprocess_state(state)
for t in count():
    # Select an action without exploration (no epsilon-greedy during evaluation)
    with torch.no_grad():
        action = policy_net(state).max(1)[1].view(1, 1)

    observation, reward, terminated, truncated, _ = env.step(action.item())
    observation = preprocess_state(observation)
    
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Check if the episode is over
    if terminated or truncated:
        break

print("Evaluation Complete")
'''