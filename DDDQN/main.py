from utils import evaluate_policy, str2bool, flatten_observation
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


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='Iudex')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training, see with tensorboard --logdir=runs')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')
parser.add_argument('--CustomReward', type=str2bool, default=True, help='Use custom reward function for Iudex env')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(10e100), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(50e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e4), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

class PreprocessedEnvWrapper(gym.Wrapper):
    def __init__(self, env, preprocess_func):
        super().__init__(env)
        self.preprocess_func = preprocess_func
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If attribute not found, forward to the base environment
            return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.preprocess_func(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.preprocess_func(obs)  # Apply preprocessing here
        return obs, reward, terminated, truncated, info

def compute_reward(game_state, next_game_state):
    """Custom reward computation logic."""
    # Reward for hitting the boss
    boss_hp_diff = game_state.boss_hp - next_game_state.boss_hp
    hit_reward = 50 * (boss_hp_diff / game_state.boss_max_hp)  # Scale up significantly, gives approx 30 of reward per hit
    #print('hit reward', hit_reward)

    # Negative Reward for getting hit
    player_hp_diff = next_game_state.player_hp - game_state.player_hp
    hit_taken_reward = 30 * (player_hp_diff / game_state.player_max_hp)  # High negative reward between 3-10 penalty

    # Penalty for rolling
    # print(game_state.player_animation)
    valid_roll = ["RollingMedium", "RollingMediumSelftra"]
    roll_penalty = -2 if game_state.player_animation in valid_roll else 0  # decent penalty for rolling 

    # Penalty for time spent
    time_penalty = -0.01  # Small penalty per step for time spent

    # huge penalty if player dies
    # experimental im nit sure if this would work -> doesnt rly lol
    death = -10 if next_game_state.player_hp == 0 else 0

    # Experimental: Reward for moving towards the arena center, no reward within 4m distance
    d_center_now = np.linalg.norm(next_game_state.player_pose[:2] - np.array([139., 596.]))
    d_center_prev = np.linalg.norm(game_state.player_pose[:2] - np.array([139., 596.]))
    move_reward = 0.1 * (d_center_prev - d_center_now) * (d_center_now > 4)

    # print(f'hit {hit_reward}, hit taken {hit_taken_reward}, roll {roll_penalty}')
    # Combine rewards and penalties
    total_reward = hit_reward + hit_taken_reward + roll_penalty + time_penalty +  move_reward # + death 
    #print(total_reward)
    return total_reward

def main():
    EnvName = ['SoulsGymIudex-v0'] 
    BriefEnvName = ['Iudex'] 

    if opt.CustomReward:
        IudexEnv.compute_reward = staticmethod(compute_reward)

    env = gym.make(EnvName[opt.EnvIdex])
    env = PreprocessedEnvWrapper(env, flatten_observation)
    opt.state_dim = 26 # env.observation_space.shape[0] # PLEASE FIX THIS LATER
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps

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
          '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '  max_e_steps:',opt.max_e_steps, '\n')

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name,BriefEnvName[opt.EnvIdex],opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel: agent.load(algo_name,BriefEnvName[opt.EnvIdex],opt.ModelIdex)

    if opt.render:
        while True:
            score = evaluate_policy(env, agent, 1)
            print('EnvName:', BriefEnvName[opt.EnvIdex], 'seed:', opt.seed, 'score:', score)
    else:
        start_time = time.time()
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            env_seed += 1
            done = False

            '''Interact & trian'''
            while not done:
                #e-greedy exploration
                if total_steps < opt.random_steps: a = env.action_space.sample()
                else: a = agent.select_action(s, deterministic=False)
                
                s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                done = (dw or tr)

                agent.replay_buffer.add(s, a, r, s_next, dw)
                s = s_next

                '''Update'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    # print('--- Training ---')
                    for j in range(opt.update_every): agent.train()

                '''Noise decay & Record & Log'''
                if total_steps % 1000 == 0: agent.exp_noise *= opt.noise_decay
                if total_steps % opt.eval_interval == 0:
                    score = evaluate_policy(env, agent, turns = 3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)

                        elapsed_time = time.time() - start_time
                        elapsed_minutes = elapsed_time / 60
                        writer.add_scalar('Time/Elapsed_Minutes', elapsed_minutes, global_step=total_steps)
                    print('EnvName:',BriefEnvName[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))
                total_steps += 1

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    print('--- Saving model ---')
                    agent.save(algo_name,BriefEnvName[opt.EnvIdex],int(total_steps/1000))
    env.close()

if __name__ == '__main__':
    main()