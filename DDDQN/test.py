from utils import flatten_observation, compute_reward
from main import PreprocessedEnvWrapper
import gymnasium as gym
import torch
from DQN import DQN_agent
import soulsgym

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
    
    agent = DQN_agent(**test_args)
    
    # 3. Load Trained Model
    model_path = "model\DuelDDQN_IudexSimpleReward_1150.pth"  # Update this path
    agent.q_net.load_state_dict(torch.load(model_path, map_location=test_args['dvc'], weights_only=True))
    agent.q_net.eval()  # Set to evaluation mode
    
    # 4. Test 
    for episode in range(5):  # Run 5 episodes for testing
        state, _ = demo_env.reset()
        done = False
        total_reward = 0
        
        while not done:            
            # Get action from policy (deterministic for testing)
            action = agent.select_action(state, deterministic=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = demo_env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
        print(f"Episode {episode+1} - Total Reward: {total_reward:.2f}")
    
    demo_env.close()

if __name__ == '__main__':
    test_model()