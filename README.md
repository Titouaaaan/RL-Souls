![Dark Souls III](https://static.bandainamcoent.eu/high/dark-souls/dark-souls-3/00-page-setup/ds3_game-thumbnail.jpg)
# RL-Souls

The goal of this project is to use Reinforcement Learning algorithms to teach an agent how to play Souls games (FromSoftware) like **Dark Souls III**.  
Different algorithms are tested and fine-tuned to figure out which methods work best.

I’m working on this project during my gap year between my M1 and M2 at Sorbonne Université.

---

## First Demo
This is a short clip of my agent Sir Rollsalot the Untrained, who at this stage has trained for about 50 hours total (phase 1 and 2 combined). Currently gets about 30% win rate as of 05/06/2025 (will be updated once he gets more training time)

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/qfs3fYU9Z3k/0.jpg)](https://www.youtube.com/watch?v=qfs3fYU9Z3k)

## ⚙️ Setup & Requirements

**This project currently only runs on Windows!**

1. **Make sure you own a copy of [Dark Souls III](https://store.steampowered.com/app/374320/DARK_SOULS_III/)** and that it is installed on your system.  
2. **Follow the SoulsGym setup instructions** from their official docs: https://soulsgym.readthedocs.io/en/latest/

3. **Create a virtual environment** so you don’t mess up your base Python install:  
   (Make sure you’re using Python **3.11.2**)

   ```bash
   py -3.11.2 -m venv myenv
   ```
4. Activate the environment
    ```
    myenv\Scripts\activate
    ```
5. Install the dependencies from the requirements file. **Little Warning**, since the repo currently contains multiple implementations, each folder contains its own requirements.txt (so ideally a dedicated virtual environment). This is not ideal and will be changed but as long as you see this message then it means I haven't changed it yet (but I will). The reason this is how it is is because of torch CUDA installations and also dependency conflicts for certains libraries (most of it is fixed though)

## Code Overview

This repo contains multiple implementations:

### ✅ DQN, Double DQN, and Dueling DQN

- **DQN (Deep Q-Network):**  
  Uses a neural network to estimate Q-values instead of a Q-table. Can generalize over high-dimensional state spaces (like what you'd get in Dark Souls).

- **Double DQN:**  
  Fixes overestimation issues in DQN by decoupling action selection and evaluation between two networks — helps stabilize learning.

- **Dueling DQN:**  
  Separates value and advantage streams to better estimate which states are valuable independently of action choice (so pretty much is my current state objectively good or not).

> These implementations didn't reach great performance — but that's most likely due to **insufficient training time** and the **high instability** of RL in such a complex environment like Dark Souls III.
> Which is why the next implementaiton performs much better:

---

### 🧠 TorchRL Implementation

This repo includes a version of **Double DQN using [TorchRL](https://pytorch.org/rl/)** — a PyTorch-native library for RL that provides components like:

- Environments  
- Data collectors  
- Replay buffers  
- Loss modules  
- Agents  

#### 📦 Pipeline Breakdown:

1. Collect experience by interacting with the SoulsGym environment (randomly at first to fill the rb).  
2. Store experiences in a replay buffer.  
3. Sample mini-batches and update the network.  
4. Periodically update the target network.

---

## 📝 PROJECT DIARY

Follow the full dev process here:  
📓 [Project Diary (Google Doc)](https://docs.google.com/document/d/1M2HvsFlbMib0nFVNFCUXnuU_41kA6tJAe7srcPvQCS0/edit?usp=sharing)
