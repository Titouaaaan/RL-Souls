from utils import flatten_observation, str2bool
from main import PreprocessedEnvWrapper
import gymnasium as gym
import torch
from DQN import DQN_agent
import soulsgym
import yaml
from main import opt
import argparse

def test_model():
    # 1. Create Demo Environment
    demo_env = gym.make("SoulsGymIudexDemo-v0")
    demo_env = PreprocessedEnvWrapper(demo_env, flatten_observation)
    
    # 2. Initialize Agent with same parameters
    test_args = {
        'dvc': 'cuda' if torch.cuda.is_available() else 'cpu',
        'state_dim': 26,
        'action_dim': demo_env.action_space.n,
        'net_width': 200,
        'lr': 1e-4,
        'Double': True,
        'Duel': True
    }
    
    agent = DQN_agent(**vars(opt))
    
    # 3. Load Trained Model
    agent.load(algo='DuelDDQN', EnvName='Iudex-v1.4-basic-reward')
    agent.q_net.eval()  # Set to evaluation mode
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

def env_test():
    from soulsgym.envs.darksouls3.iudex import IudexEnv
    env = IudexEnv(
        game_speed=1,
        phase=2,
        init_pose_randomization=False
    )
    obs, info = env.reset()
    terminated = False
    while not terminated:
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()

def change_action_space():
    env = gym.make("SoulsGymIudex-v0")

    file = r"D:\GAP YEAR\RL-Souls\DDDQN\dddqnvenv\Lib\site-packages\soulsgym\core\data\darksouls3\actions.yaml"

    with open(r"D:\GAP YEAR\RL-Souls\DDDQN\dddqnvenv\Lib\site-packages\soulsgym\core\data\darksouls3\actions.yaml", "r") as file:
        action_mapping = yaml.safe_load(file)

    new_action_space = {
        0: ['forward'], 1: ['forward', 'right'], 2: ['right'], 3: ['right', 'backward'], 4: ['backward'], 
        5: ['backward', 'left'], 6: ['left'], 7: ['left', 'forward'], 8: ['forward', 'roll'],  
        9: ['right', 'roll'], 10: ['backward', 'roll'],  11: ['left', 'roll'], 
         12: ['lightattack'], 13: ['heavyattack'], 14: ['parry'], 15: []}
    
    with open(r"D:\GAP YEAR\RL-Souls\DDDQN\dddqnvenv\Lib\site-packages\soulsgym\core\data\darksouls3\actions.yaml", 'w') as file:
        yaml.dump(new_action_space, file, default_flow_style=False)

    #env.action_space = new_action_space
    env.action_space = gym.spaces.Discrete(15)
    obs, info = env.reset()
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        action_name = " + ".join(action_mapping.get(action, ["Unknown Action"]))
        print('action: ', action_name)
        next_obs, reward, terminated, truncated, info = env.step(action)
    env.close()

