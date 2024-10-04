import gymnasium
import soulsgym
import time
import threading
import soulsgym.core
from soulsgym.core.game_input import GameInput
import soulsgym.core.static

stop_spamming = threading.Event()
game_id = 'DarkSoulsIII'  # Replace with the correct game ID if needed

def spam_q_key(game_id: str):
    game_input = GameInput(game_id=game_id)
    while not stop_spamming.is_set():
        # Simulate pressing 'Q' key (or lock-on in Dark Souls)
        print(soulsgym.core.static.keybindings)
        game_input.add_action('lockon')  # Lock-on action
        game_input.update_input()
        print('pressed q?')
        time.sleep(2)  # Sleep for a short time to spam at intervals

if __name__ == "__main__":
    # Start the thread for spamming the 'Q' key
    q_spam_thread = threading.Thread(target=spam_q_key, args=(game_id,))
    q_spam_thread.daemon = True
    q_spam_thread.start()

    # Wait a short time to allow spamming to begin
    time.sleep(2)  # Adjust this if needed, depending on the game

    # Now reset the environment (Q key spam should already be happening)
    env = gymnasium.make("SoulsGymIudex-v0", autoreset=True)
    obs, info = env.reset()
    print('env has been reset, stopping thread')
    stop_spamming.set()
    q_spam_thread.join()  # Ensure the thread finishes before closing the environment
    terminated = False

    # Main loop for running the environment
    while not terminated:
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    # Stop spamming thread once environment is done
    env.close()
