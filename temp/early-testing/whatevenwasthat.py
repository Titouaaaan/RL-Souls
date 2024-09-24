import gymnasium
import soulsgym
import time
import threading
from soulsgym.core.game_input import GameInput

""" stop_spamming = threading.Event()

def spam_q_key(game_id: str):
    game_input = GameInput(game_id=game_id)
    while not stop_spamming.is_set():
        game_input.add_action(action='lockon')
        game_input.update_input() """

if __name__ == "__main__":
    # Create the game environment
    env = gymnasium.make("SoulsGymIudex-v0")
    obs, info = env.reset()
    """ stop_spamming.set() """
    terminated = False

    """ # Get the correct game_id for GameInput
    game_id = 'DarkSoulsIII'  # Replace with the correct game ID if needed

    # Start the thread for spamming the 'Q' key
    q_spam_thread = threading.Thread(target=spam_q_key, args=(game_id,))
    q_spam_thread.daemon = True  # Ensures thread exits when the main program ends
    q_spam_thread.start() """

    # Main loop for running the environment
    while not terminated:
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    env.close()
