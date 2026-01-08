# RL Fundamentals 🤖

This repository contains foundational **Reinforcement Learning (RL)** implementations developed as part of my research training at the [EXACT Lab](https://exact.umd.edu/) at UMD.

The goal of this collection is to demonstrate proficiency in both tabular methods and Deep Reinforcement Learning (DRL) using practice standard environments, with the goal of applying these techniques to dexterous robotic manipulation.

## Repository Structure

### 1. Q-Learning (Tabular)
Located in `q_learning/`:

* **`frozen_lake.py`**: A from-scratch implementation of the Q-Learning algorithm.
    * **Features:** Custom epsilon-greedy exploration with decay, moving average reward visualization, and training error tracking.
    * **Environment:** `FrozenLake-v1` (Discrete state space).
* **`random_agent.py`**: A baseline agent used to benchmark environment complexity before applying learning algorithms.

### 2. Proximal Policy Optimization (PPO)
Located in `ppo/`:

* **`cartpole_ppo.py`**: Solves the classic CartPole balancing problem.
    * **Technique:** Uses an **MlpPolicy** (Multi-Layer Perceptron) for vector-based observation spaces.
    * **Framework:** Stable-Baselines3.
* **`carracing_ppo.py`**: An agent trained to drive a race car from raw pixel inputs.
    * **Technique:** Uses a **CnnPolicy** (Convolutional Neural Network) to extract features from image frames (RGB states).
    * **Environment:** `CarRacing-v3`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arikgershman/rl_fundamentals.git
   cd rl_fundamentals
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Running the Q-Learning Agent:**
```bash
python q_learning/frozen_lake.py
```
*Output: Displays training progress bar (tqdm) and generates plots for Reward, Episode Length, and Training Error.*

**Running PPO (Car Racing):**
```bash
python ppo/carracing_ppo.py
```
*Output: Trains the agent for 25k timesteps and renders the result.*

## Context

These implementations serve as preliminary work for broader research into robot learning and risk analysis conducted at the [EXACT Lab](https://exact.umd.edu/). The focus is on understanding the transition from discrete state-space planning to high-dimensional continuous control.
