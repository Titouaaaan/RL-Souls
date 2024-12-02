import gymnasium
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim
from gymnasium.spaces.utils import flatten
import soulsgym

from soulsgym.envs.darksouls3.iudex import IudexEnv

import numpy as np
import random
import collections
import csv
import os
from datetime import datetime
import time

import soulsgym.core
import soulsgym.core.speedhack
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
) 

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., "20231126_150102"
run_folder = f"run_{timestamp}"
os.makedirs(run_folder, exist_ok=True)

# File paths for saving logs, models, and parameters
performance_file = os.path.join(run_folder, "performance.csv")
model_file = os.path.join(run_folder, "model.pth")
optimizer_file = os.path.join(run_folder, "optimizer.pth")
params_file = os.path.join(run_folder, "params.txt")


class QNetwork(nn.Module):
    """
    Deep Q-Network (DQN) Architecture.

    This architecture is designed to approximate the Q-function for a reinforcement learning agent.
    The choice of architecture considers the complexity of the task, the size of the observation
    space (110 features after flattening), and the need for robust and stable training. The main 
    architectural decisions are as follows:

    - **Fully Connected Layers**: Suitable for handling flattened, non-sequential observations.
      The architecture includes three hidden layers to capture complex relationships in the state space.

    - **Hidden Dimension**: Each hidden layer uses a configurable number of neurons (`hidden_dim`), 
      which provides sufficient capacity for learning without overfitting. Larger values (e.g., 128 or 256) 
      are recommended for larger observation spaces.

    - **Leaky ReLU Activation**: Used to prevent the "dying ReLU" problem and ensure gradient flow even 
      for small or negative activations. This is especially useful in deeper networks.

    - **Dropout Regularization**: Prevents overfitting by randomly deactivating neurons during training. 
      A small dropout probability (e.g., 0.2) balances regularization without excessive information loss.

    - **Output Layer**: The final layer outputs Q-values for all possible actions, with no activation function. 
      This allows the Q-values to remain unbounded, as required by the Bellman equation.

    Args:
        action_dim (int): Number of possible actions in the action space.
        state_dim (int): Dimensionality of the flattened observation space.
        hidden_dim (int): Number of neurons in each hidden layer.

    Returns:
        torch.Tensor: Predicted Q-values for all actions, given an input state.
    """
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()
        # if starting hidden_dim is 64 we get 64 -> 128 -> 256
        self.fc_1 = nn.Linear(state_dim, hidden_dim) 
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc_3 = nn.Linear(hidden_dim*2, hidden_dim*4)
        self.fc_out = nn.Linear(hidden_dim*4, action_dim)

        self.dropout = nn.Dropout(0.2)  # Regularization

    def forward(self, inp):
        # Input -> Hidden Layer 1
        x = F.leaky_relu(self.fc_1(inp))
        x = self.dropout(x)  # Dropout after activation

        # Hidden Layer 2
        x = F.leaky_relu(self.fc_2(x))
        x = self.dropout(x)

        # Hidden Layer 3
        x = F.leaky_relu(self.fc_3(x))
        x = self.dropout(x)

        # Output Layer
        x = self.fc_out(x)

        return x

class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state) #do state.cpu().numpy() maybe for consistency?
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        """ state = np.array(self.state)
        action = np.array(self.action)
        return torch.Tensor(state)[idx].to(device), torch.LongTensor(action)[idx].to(device), \
               torch.Tensor(state)[1+np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device) """
        # Convert deque to list and then stack tensors
        states = torch.stack(list(self.state)).to(device)
        actions = torch.tensor(self.action, dtype=torch.long).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        is_done = torch.tensor(self.is_done, dtype=torch.float32).to(device)

        # Next states
        next_states = states[1 + torch.tensor(idx)].to(device)

        return (
            states[idx],         # Sampled states
            actions[idx],        # Corresponding actions
            next_states,         # Corresponding next states
            rewards[idx],        # Corresponding rewards
            is_done[idx],        # Done flags
        )

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()

def select_action(model, env, state, eps):
    state = torch.Tensor(state).to(device)
    with torch.no_grad():
        values = model(state)

    # select a random action wih probability eps
    if random.random() <= eps:
        action = np.random.randint(0, env.action_space.n)
    else:
        action = np.argmax(values.cpu().numpy())

    return action


def train(batch_size, current, target, optim, memory, gamma):
    #print('entering train!')
    states, actions, next_states, rewards, is_done = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - is_done)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    #print(f'loss: {loss}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    #print('leaving train')


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    print('evaluating eval')
    Qmodel.eval()
    perform = 0
    for n_rep in range(repeats):
        #print(f'Eval episode #{n_rep}') #debugging
        state, info = env.reset()
        #convert the obs into a tensor to avoid dealing with dicts
        state = torch.tensor(flatten(env.observation_space, state), dtype=torch.float32).to(device)
        done = False
        while not done:
            # state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, truncated, _ = env.step(action)
            state = torch.tensor(flatten(env.observation_space, state), dtype=torch.float32).to(device)
            perform += reward
    Qmodel.train()
    #print('exiting eval')
    return perform/repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

observation_space = spaces.Dict({
    "phase": spaces.Discrete(2),  # Assuming 5 possible phases
    "player_hp": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),  # Adjust max HP as needed
    "player_max_hp": spaces.Box(low=1, high=1000, shape=(), dtype=np.int32),
    "player_sp": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
    "player_max_sp": spaces.Box(low=1, high=1000, shape=(), dtype=np.int32),
    "boss_hp": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),  # Supports single or multi-boss fights
    "boss_max_hp": spaces.Box(low=1, high=1000, shape=(1,), dtype=np.int32),
    "player_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
    "boss_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
    "camera_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
    "player_animation_duration": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
    "boss_animation_duration": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),  # Adjust if multi-boss
    "lock_on": spaces.Discrete(2)  # Boolean as 0 or 1
})

def game_state_to_observation(game_state):
    return {
        "phase": game_state.phase,
        "player_hp": game_state.player_hp,
        "player_max_hp": game_state.player_max_hp,
        "player_sp": game_state.player_sp,
        "player_max_sp": game_state.player_max_sp,
        "boss_hp": np.array([game_state.boss_hp] if isinstance(game_state.boss_hp, int) else game_state.boss_hp),
        "boss_max_hp": np.array([game_state.boss_max_hp] if isinstance(game_state.boss_max_hp, int) else game_state.boss_max_hp),
        "player_pose": game_state.player_pose,
        "boss_pose": game_state.boss_pose,
        "camera_pose": game_state.camera_pose,
        "player_animation_duration": game_state.player_animation_duration,
        "boss_animation_duration": (
            np.array([game_state.boss_animation_duration])
            if isinstance(game_state.boss_animation_duration, float)
            else game_state.boss_animation_duration
        ),
        "lock_on": int(game_state.lock_on),  # Convert boolean to integer
    }

def compute_reward(game_state, next_game_state):
    """Custom reward computation logic."""
    # Reward for hitting the boss
    boss_hp_diff = game_state.boss_hp - next_game_state.boss_hp
    hit_reward = 10 * (boss_hp_diff / game_state.boss_max_hp)  # Scale up significantly

    # Reward for getting hit
    player_hp_diff = next_game_state.player_hp - game_state.player_hp
    hit_taken_reward = 5 * (player_hp_diff / game_state.player_max_hp)  # High positive reward

    # Penalty for rolling
    # print(game_state.player_animation)
    valid_roll = ["RollingMedium", "RollingMediumSelftra"]
    roll_penalty = -0.5 if game_state.player_animation in valid_roll else 0  # Small penalty for rolling

    # Penalty for time spent
    time_penalty = -0.01  # Small penalty per step for time spent

    # Combine rewards and penalties
    total_reward = hit_reward + hit_taken_reward + roll_penalty + time_penalty

    return total_reward

IudexEnv.compute_reward = staticmethod(compute_reward) #update the reward function of the library to our custom one

def main(gamma=0.99, lr=5e-4, min_episodes=20, eps=1, eps_decay=0.995, eps_min=0.01, update_step=50, batch_size=128, update_repeats=30,
         num_episodes=10000, seed=42, max_memory_size=20000, lr_gamma=1, lr_step=100, measure_step=100,
         measure_repeats=100, hidden_dim=64, env_name='SoulsGymIudex-v0', save_model=50, cnn=False, horizon=np.inf, render=False, render_step=50):
    """
    Remark: Convergence is slow. Wait until around episode 2500 to see good performance.

    :param gamma: reward discount factor
    :param lr: learning rate for the Q-Network
    :param min_episodes: we wait "min_episodes" many episodes in order to aggregate enough data before starting to train
    :param eps: probability to take a random action during training
    :param eps_decay: after every episode "eps" is multiplied by "eps_decay" to reduces exploration over time
    :param eps_min: minimal value of "eps"
    :param update_step: after "update_step" many episodes the Q-Network is trained "update_repeats" many times with a
    batch of size "batch_size" from the memory.
    :param batch_size: see above
    :param update_repeats: see above
    :param num_episodes: the number of episodes played in total
    :param seed: random seed for reproducibility
    :param max_memory_size: size of the replay memory
    :param lr_gamma: learning rate decay for the Q-Network
    :param lr_step: every "lr_step" episodes we decay the learning rate
    :param measure_step: every "measure_step" episode the performance is measured
    :param measure_repeats: the amount of episodes played in to asses performance
    :param hidden_dim: hidden dimensions for the Q_network
    :param env_name: name of the gym environment
    :param save_model: after every save_model episodes, save the model
    :param cnn: set to "True" when using environments with image observations like "Pong-v0"
    :param horizon: number of steps taken in the environment before terminating the episode (prevents very long episodes)
    :param render: if "True" renders the environment every "render_step" episodes
    :param render_step: see above
    :return: the trained Q-Network and the measured performances
    """
    params = { #im saving it here even tho its not the best but ill fix it later
            "gamma": gamma,
            "lr": lr,
            "min_episodes": min_episodes,
            "eps": eps,
            "eps_decay": eps_decay,
            "eps_min": eps_min,
            "update_step": update_step,
            "batch_size": batch_size,
            "update_repeats": update_repeats,
            "num_episodes": num_episodes,
            "seed": seed,
            "max_memory_size": max_memory_size,
            "lr_step": lr_step,
            "lr_gamma": lr_gamma,
            "measure_step": measure_step,
            "measure_repeats": measure_repeats,
            "hidden_dim": hidden_dim,
            "env_name": env_name,
            "save_model": save_model,
            "cnn": cnn,
            "horizon": horizon
            }
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = gymnasium.make(env_name)
    #env.seed(seed)
    state, info = env.reset()
    state = torch.tensor(flatten(env.observation_space, state), dtype=torch.float32).to(device)
    """ print(type(state))
    print(env.action_space.n)
    print(env.observation_space) """
    state_dim = flatdim(env.observation_space)
    """ print("Flattened state dimension:", state_dim) """

    
    Q_1 = QNetwork(action_dim=env.action_space.n, state_dim=state_dim,
                       hidden_dim=hidden_dim).to(device)
    Q_2 = QNetwork(action_dim=env.action_space.n, state_dim=state_dim,
                       hidden_dim=hidden_dim).to(device)
    # transfer parameters from Q_1 to Q_2
    update_parameters(Q_1, Q_2)

    # we only train Q_1
    for param in Q_2.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Q_1.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    memory = Memory(max_memory_size)
    performance = []
    avg_reward = []
    begin_episodes = time.time()
    for episode in range(num_episodes):
        # display the performance
        if (episode % measure_step == 0) and episode >= min_episodes:
            perf = [episode, evaluate(Q_1, env, measure_repeats)]
            performance.append(perf)
            print("Episode: ", episode)
            print("rewards: ", performance[-1][1])
            print("lr: ", scheduler.get_last_lr()[0])
            print("eps: ", eps)

            with open(performance_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(perf)
                print('Data logged!')

        state, info = env.reset()
        """ print('STATE DEBUG')
        print(type(state))
        print(state) """
        state = torch.tensor(flatten(env.observation_space, state), dtype=torch.float32).to(device)
        memory.state.append(state)

        done = False
        i = 0
        reward_cumsum = 0
        while not done:
            i += 1
            action = select_action(Q_2, env, state, eps)
            # print("action chose: ", action)
            state, reward, done, truncated, _ = env.step(action)
            state = torch.tensor(flatten(env.observation_space, state), dtype=torch.float32).to(device)

            #print("reward: ", reward)
            reward_cumsum += reward
            
            if i > horizon:
                done = True


            # save state, action, reward sequence
            memory.update(state, action, reward, done)
        avg_reward.append(reward_cumsum)
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"episode {episode}/{num_episodes} lasted {i} steps, cumsum reward = {reward_cumsum} (avg = {sum(avg_reward)/len(avg_reward)})")

        if episode >= min_episodes and episode % update_step == 0:
            print('Entering the training sequence!')
            for _ in range(update_repeats):
                train(batch_size, Q_1, Q_2, optimizer, memory, gamma)

            # transfer new parameter from Q_1 to Q_2
            update_parameters(Q_1, Q_2)

        # update learning rate and eps
        if episode > min_episodes:
            scheduler.step()
        eps = max(eps*eps_decay, eps_min)

        """ episode_loss.append(reward_cumsum)
        plot_loss(episode_loss, show_result=True) """

        if episode % save_model == 0: 
            torch.save(Q_1.state_dict(), model_file)

            torch.save(optimizer.state_dict(), optimizer_file)
            
            with open(params_file, 'w') as f:
                for key, value in params.items():
                    f.write(f"{key}: {value}\n")

            print('Logged data!')

    return Q_1, performance


if __name__ == '__main__':
    Q_1, performance = main()
    print(f"All run info saved in the folder {run_folder}")