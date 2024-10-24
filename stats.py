import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_log.csv')

# Usually we might need to clean up the data for anomalies (usually the first couple of episodes idk why)
# to run: 
# # C:/Users/titou/AppData/Local/Programs/Python/Python311/python.exe 'D:\GAP YEAR\RL-Souls\stats.py'
# we do this to not have conflicts, but in the future i can make a seperate virtual env to not break dependencies

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot steps per episode
axs[0].plot(df['Episode'], df['Steps'], marker='o')
axs[0].set_title('Steps per Episode')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Steps')

# Plot average reward per episode
axs[1].plot(df['Episode'], df['Avg Reward'], marker='o', color='orange')
axs[1].set_title('Average Reward per Episode')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Avg Reward')

# Plot average loss per episode
axs[2].plot(df['Episode'], df['Avg Loss'], marker='o', color='green')
axs[2].set_title('Average Loss per Episode')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Avg Loss')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()