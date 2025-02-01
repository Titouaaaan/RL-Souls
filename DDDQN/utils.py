import numpy as np

def evaluate_policy(env, agent, turns = 10):
    # print('--- Evaluating ---')
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

def compute_reward(game_state, next_game_state, hit_given_var=80, hit_taken_var=90, roll_penalty_var=-0.05, time_penalty_var=0.0, death_var=0, move_reward_var=0.5):
    """Custom reward computation logic."""
    # Reward for hitting the boss
    boss_hp_diff = game_state.boss_hp - next_game_state.boss_hp
    hit_reward = hit_given_var * (boss_hp_diff / game_state.boss_max_hp)  # Scale up significantly
    # if hit_reward != 0: print('hit reward', hit_reward)

    # Negative Reward for getting hit
    player_hp_diff = next_game_state.player_hp - game_state.player_hp
    hit_taken_reward = hit_taken_var * (player_hp_diff / game_state.player_max_hp)  
    #if hit_taken_reward != 0: print('hit reward', hit_taken_reward)

    # Penalty for rolling
    # print(game_state.player_animation)
    valid_roll = ["RollingMedium", "RollingMediumSelftra"]
    roll_penalty = roll_penalty_var if game_state.player_animation in valid_roll else 0  # small penalty for rolling 

    # Penalty for time spent
    time_penalty = time_penalty_var  # Small penalty per step for time spent

    # huge penalty if player dies
    # experimental im nit sure if this would work -> doesnt rly lol
    death = death_var if next_game_state.player_hp == 0 else 0

    # Experimental: Reward for moving towards the arena center, no reward within 4m distance
    d_center_now = np.linalg.norm(next_game_state.player_pose[:2] - np.array([139., 596.]))
    d_center_prev = np.linalg.norm(game_state.player_pose[:2] - np.array([139., 596.]))
    move_reward = move_reward_var * (d_center_prev - d_center_now) * (d_center_now > 4)

    # print(f'hit {hit_reward}, hit taken {hit_taken_reward}, roll {roll_reward}')
    # Combine rewards and penalties
    total_reward = hit_reward + hit_taken_reward + move_reward + roll_penalty + death  + time_penalty
    # print(f'hit_reward: {hit_reward} \nhit_taken_reward: {hit_taken_reward} \nmove_reward: {move_reward} \nroll_penalty: {roll_penalty} \ndeath: {death} \ntime_penalty: {time_penalty}')
    # print('total: ', total_reward)
    #print(total_reward)
    return total_reward # maybe divide by large num lke 100 if rewards are too big or too small