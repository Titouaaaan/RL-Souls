import numpy as np

def evaluate_policy(env, agent, turns = 10):
    print('--- Evaluating ---')
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    env.reset()
    return int(total_scores/turns)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise

def flatten_observation(obs):
    """
    Convert the custom observation dictionary into a 1D tensor.
    """
    # Flatten scalar values (convert to float)
    scalar_values = [
        obs['phase'],
        obs['player_hp'][0],
        obs['player_max_hp'],
        obs['player_sp'][0],
        obs['player_max_sp'],
        obs['boss_hp'][0],
        obs['boss_max_hp'],
        obs['player_animation'],
        obs['player_animation_duration'][0],
        obs['boss_animation'],
        obs['boss_animation_duration'][0],
        int(obs['lock_on'])  # Convert boolean to int
    ]

    # Flatten arrays
    pose_values = np.concatenate([
        obs['player_pose'],
        obs['boss_pose'],
        obs['camera_pose']
    ])

    flat_obs = np.concatenate([scalar_values, pose_values])

    return flat_obs