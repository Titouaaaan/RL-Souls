import gymnasium
import soulsgym


if __name__ == "__main__":
    env = gymnasium.make("SoulsGymIudex-v0")
    obs, info = env.reset()
    print('Initial Obs:\n', obs, '\nInitial Info:\n', info)
    terminated = False
    while not terminated:
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print('\nNext Obs: \n', next_obs, '\nReward:\n', reward)
    env.close()