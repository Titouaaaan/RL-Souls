import gymnasium
import soulsgym
import logging


if __name__ == "__main__":
    # Set log level to DEBUG
    soulsgym.set_log_level(level=logging.DEBUG)

    # Create a new environment and start the random agent
    print('Starting new environment...')
    env = gymnasium.make("SoulsGymIudex-v0")
    print('Environment created')

    obs, info = env.reset()
    print('base observations: ', obs)
    print('Environment reset')

    terminated = False
    while not terminated:
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print('Next obs: ', next_obs)

    # Close the environment at the end
    env.close()
    print('Environment closed')