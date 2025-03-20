from utils import evaluate_policy, str2bool, flatten_observation, compute_reward, compute_reward_basic
from datetime import datetime
import time
from soulsgym.envs.darksouls3.iudex import IudexEnv
from DQN import DQN_agent
import gymnasium as gym
import numpy as np
import os, shutil
import argparse
import torch
import soulsgym
from main import PreprocessedEnvWrapper
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='Iudex')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training, see with tensorboard --logdir=runs')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--CustomReward', type=str2bool, default=True, help='Use custom reward function for Iudex env')

parser.add_argument('--hitGivenVar', type=int, default=50, help='reward multiplier for dealing damage')
parser.add_argument('--hitTakenVar', type=int, default=50, help='reward multiplier for taking damage')
parser.add_argument('--rollPenalty', type=float, default=-0.5, help='roll penalty value')
parser.add_argument('--timePenalty', type=float, default=0.0, help='penalty for stalling (stay alive too long)')
parser.add_argument('--deathPenalty', type=int, default=0, help='penalty for dying (experimental)')
parser.add_argument('--moveReward', type=float, default=0.5, help='reward for moving to center of arena (experimental)')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e10), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e4), help='Model evaluating interval, in steps.')
parser.add_argument('--eval_turns', type=int, default=3, help='How many episodes for eval')
parser.add_argument('--random_steps', type=int, default=int(5e4), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=1000, help='training frequency')
parser.add_argument('--eps_decay_rate', type=int, default=3000, help='decay rate every n episodes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=128, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=400, help='lenth of sliced trajectory')
parser.add_argument('--epsilon', type=float, default=1.0, help='eps for e greedy strategy')
parser.add_argument('--epsilon_decay', type=float, default=0.999, help='decay rate of exploration')
parser.add_argument('--epsilon_min', type=float, default=0.1, help='min value for e greedy eps value')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)


EnvName = ['SoulsGymIudexDemo-v0'] # SoulsGymIudexDemo-v0 => for full fight to test out agent
BriefEnvName = ['Iudex-v2.0-reduced-action-space'] #

if opt.CustomReward:
    # wrapper to add our new parameters to the reward function while still being able to access the game state and next game state
    def reward_wrapper(game_state, next_game_state):
        return compute_reward_basic(
            game_state=game_state,
            next_game_state=next_game_state
        )
    
    IudexEnv.compute_reward = staticmethod(reward_wrapper)

demo_env = gym.make(EnvName[opt.EnvIdex])
new_action_space = {
        0: ['forward'], 1: ['forward', 'right'], 2: ['right'], 3: ['right', 'backward'], 4: ['backward'], 
        5: ['backward', 'left'], 6: ['left'], 7: ['left', 'forward'], 8: ['forward', 'roll'],  
        9: ['right', 'roll'], 10: ['backward', 'roll'],  11: ['left', 'roll'], 
         12: ['lightattack'], 13: ['heavyattack'], 14: ['parry'], 15: []}
    
# REPLACE WITH YOUR OWN VENV PATH THIS IS ONLY TEMPORARY 
# I KNOW IT LOOKS HORRIBLE BUT I WILL CHANGE IT LATER
with open(r"D:\GAP YEAR\RL-Souls\DDDQN\dddqnvenv\Lib\site-packages\soulsgym\core\data\darksouls3\actions.yaml", 'w') as file:
    yaml.dump(new_action_space, file, default_flow_style=False)

#env.action_space = new_action_space
demo_env.action_space = gym.spaces.Discrete(15)

demo_env = PreprocessedEnvWrapper(demo_env, flatten_observation)
opt.state_dim = 26 # env.observation_space.shape[0] # PLEASE FIX THIS LATER
opt.action_dim = demo_env.action_space.n
opt.max_e_steps = demo_env._max_episode_steps # remopve this in case we use the IudexEnv() method above (change line 129 too)

#Algorithm Setting
if opt.Duel: algo_name = 'Duel'
else: algo_name = ''
if opt.Double: algo_name += 'DDQN'
else: algo_name += 'DQN'

# Seed Everything
env_seed = opt.seed
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Random Seed: {}".format(opt.seed))

print('Algorithm:',algo_name,'  Env:',BriefEnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,
        '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, ' max_e_steps:', opt.max_e_steps,'\n') # remove in case of env creation change

agent = DQN_agent(**vars(opt))

load_algo_name = 'DuelDDQN'
load_env_name = 'Iudex-v2.0-reduced-action-space'
resumed_steps = agent.load(load_algo_name, load_env_name, checkpoint=True)
agent.replay_buffer.load(f"./model/{load_algo_name}_{load_env_name}_replaybuf.pth")
agent.q_net.eval()

file = r"D:\GAP YEAR\RL-Souls\DDDQN\dddqnvenv\Lib\site-packages\soulsgym\core\data\darksouls3\actions.yaml"
with open(file, "r") as file:
    action_mapping = yaml.safe_load(file)

# 4. Test 
for episode in range(5):  # Run 5 episodes for testing
    state, _ = demo_env.reset()
    done = False
    total_reward = 0
    
    while not done:            
        # Get action from policy (deterministic for testing)
        action = agent.select_action(state, deterministic=True)
        action_name = " + ".join(action_mapping.get(action, ["Unknown Action"]))
        print('action: ', action_name)
        # Take step
        next_state, reward, terminated, truncated, _ = demo_env.step(action)
        done = terminated or truncated
        
        state = next_state
        total_reward += reward
        
    print(f"Episode {episode+1} - Total Reward: {total_reward:.2f}")

demo_env.close()