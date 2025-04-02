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
import yaml

print(f'CUDA available: {torch.cuda.is_available()}')
'''Hyperparameter Setting'''
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
parser.add_argument('--update_every', type=int, default=5000, help='training frequency')
parser.add_argument('--eps_decay_rate', type=int, default=3000, help='decay rate every n episodes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=2048, help='lenth of sliced trajectory')
parser.add_argument('--epsilon', type=float, default=1.0, help='eps for e greedy strategy')
parser.add_argument('--epsilon_decay', type=float, default=0.9999, help='decay rate of exploration')
parser.add_argument('--epsilon_min', type=float, default=0.1, help='min value for e greedy eps value')

parser.add_argument('--Double', type=str2bool, default=True, help='Whether to use Double Q-learning')
parser.add_argument('--Duel', type=str2bool, default=True, help='Whether to use Duel networks')
parser.add_argument('--Enhanced', type=str2bool, default=True, help='Whether to use Enhanced Duel networks')

parser.add_argument('--debugging', type=str2bool, default=False, help='Whether to print values during training for debugging')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device

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

def main():
    EnvName = ['SoulsGymIudex-v0'] # SoulsGymIudexDemo-v0 => for full fight to test out agent
    BriefEnvName = ['Iudex-v2.3-redAS-newNet'] #

    if opt.CustomReward:
        # wrapper to add our new parameters to the reward function while still being able to access the game state and next game state
        def reward_wrapper(game_state, next_game_state):
            return compute_reward_basic(
                game_state=game_state,
                next_game_state=next_game_state
            )
            return compute_reward(
                game_state,
                next_game_state,
                hit_given_var=opt.hitGivenVar,
                hit_taken_var=opt.hitTakenVar,
                roll_penalty_var=opt.rollPenalty,
                time_penalty_var=opt.timePenalty,
                death_var=opt.deathPenalty,
                move_reward_var=opt.moveReward
            )
        
        IudexEnv.compute_reward = staticmethod(reward_wrapper)
    
    # Wanted to try this to change phase and game speed but when i do this the agent behaves in a very weird way
    # Agent starts stuttering and acting very differently and idk why
    """ env = IudexEnv(
        game_speed=4,
        phase=1,
        init_pose_randomization=False
    ) """
    env = gym.make(EnvName[opt.EnvIdex])

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
    env.action_space = gym.spaces.Discrete(15)

    env = PreprocessedEnvWrapper(env, flatten_observation)
    opt.state_dim = 26 # env.observation_space.shape[0] # PLEASE FIX THIS LATER
    opt.action_dim = env.action_space.n
    opt.max_e_steps = env._max_episode_steps # remopve this in case we use the IudexEnv() method above (change line 129 too)

    #Algorithm Setting
    if opt.Duel: algo_name = 'Duel'
    else: algo_name = ''
    if opt.Enhanced: algo_name += 'Enhanced'
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

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}-{}_S{}_'.format(algo_name,BriefEnvName[opt.EnvIdex],opt.seed) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

        # Log hyperparameters as text
        param_str = "| Param | Value |\n|-------|-------|\n"
        for arg in vars(opt):
            param_str += f"| {arg} | {getattr(opt, arg)} |\n"
        writer.add_text("Hyperparameters", param_str, 0)

        # Alternatively log as individual scalars (at step 0)
        """ for arg in vars(opt):
            writer.add_scalar(f"Parameters/{arg}", getattr(opt, arg), 0) """

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel: 
        ''' CHANGE MODEL HERE IF NEEDED '''
        print('Loading model')
        load_algo_name = 'DuelDDQN'
        load_env_name = 'Iudex-v1.0'
        resumed_steps = agent.load(load_algo_name, load_env_name, checkpoint=True)
        agent.replay_buffer.load(f"./model/{load_algo_name}_{load_env_name}_replaybuf.pth")
        total_steps = resumed_steps
        print(f'Resuming at step: {total_steps}')
    else:
        print('Training model from scratch')
        total_steps = 0
        """ agent.save(algo_name, BriefEnvName[opt.EnvIdex], checkpoint=True)
        agent.replay_buffer.save(f"./model/{algo_name}_{BriefEnvName[opt.EnvIdex]}_replaybuf.pth") """

    print(opt)
    start_time = time.time()
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

            """ if opt.debugging:
                print('reward: ', r) """

            '''Update'''
            if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                total_loss = 0.0
                for j in range(opt.update_every): 
                    loss = agent.train()
                    total_loss += loss
                avg_loss = total_loss / opt.update_every

                # if opt.debugging:
                # print('Avg loss of training: ', avg_loss)

                if opt.write:
                    writer.add_scalar('loss', avg_loss, total_steps)

            '''Epsilon decay & Record & Log'''
            ''' Move this to after every episode?'''
            """ if total_steps % opt.eps_decay_rate == 0: 
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)  """

            if total_steps % opt.eval_interval == 0:
                score = evaluate_policy(env, agent, turns = opt.eval_turns)
                if opt.write:
                    writer.add_scalar('ep_r', score, global_step=total_steps)
                    writer.add_scalar('e-greedy eps', agent.epsilon, global_step=total_steps)

                    elapsed_time = time.time() - start_time
                    elapsed_minutes = elapsed_time / 60
                    writer.add_scalar('Time/Elapsed_Minutes', elapsed_minutes, global_step=total_steps)
                print(
                    'EnvName:',BriefEnvName[opt.EnvIdex],
                    'seed:',opt.seed,
                    'steps: {}k'.format(int(total_steps/1000)),
                    'score:', score, 
                    'time elapsed:', elapsed_minutes,
                    'eps: ', agent.epsilon
                    )
            

            """ if total_steps % 5000 == 0:
                agent.q_target.load_state_dict(agent.q_net.state_dict()) """

            '''save model'''
            if total_steps % opt.save_interval == 0:
                if opt.debugging: 
                    print('--- Saving model ---')
                agent.save(algo_name, BriefEnvName[opt.EnvIdex], steps=total_steps, checkpoint=True)
                agent.replay_buffer.save(f"./model/{algo_name}_{BriefEnvName[opt.EnvIdex]}_replaybuf.pth")
            total_steps += 1

        # Epsilon decay after episodes instead of steps?
        if total_steps >= 5e4:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    env.close()

if __name__ == '__main__':
    main()