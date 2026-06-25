# Traffic Signal Control using Reinforcement Learning

This repository implements three model-free reinforcement learning (RL) algorithms—Q-Learning, SARSA, and Deep Q-Networks (DQN) .to optimize traffic signal switching at a simulated four-way intersection. By framing the problem as a Markov Decision Process (MDP), the RL agents learn to adapt to dynamic traffic patterns, outperforming traditional rule-based controllers and significantly reducing vehicle wait times.

## The Environment

The intersection simulation runs for 300 steps and models dynamic traffic demands, mimicking morning, mid-day, and evening rush hours with varying vehicle spawn rates.

* **State Space:** Tracks the number of waiting cars in the North/South and East/West directions (capped at 15) and the current traffic light phase. This creates a total of 1,024 possible states.
* **Action Space:** A binary choice at each step to either maintain the current green light (0) or initiate a phase switch (1). A minimum 3-step green-time constraint is enforced to prevent rapid, inefficient toggling.
* **Reward Function:** Designed to minimize traffic jams. The primary model penalizes the total number of waiting vehicles per step. 

## Core Algorithms & Baselines

We implemented and evaluated three RL agents, training them from scratch without hard-coded traffic rules:
* **Q-Learning:** Off-policy TD control for aggressive optimal policy learning.
* **SARSA:** On-policy TD control for cautious, exploration-aware learning.
* **Deep Q-Network (DQN):** Neural network function approximation (6 -> 64 -> 64 -> 2) utilizing Experience Replay and a Target Network for stability.

These agents are compared against standard, non-learning baselines: Fixed-Time timers (5-step and 10-step), Random Switch, and Longest Queue First (LQF).

## Performance Results

After training (20,000 episodes for tabular methods, 2,500 for DQN), the models were evaluated over 200 greedy episodes. The RL models consistently outperformed all non-learning heuristics.

| Strategy | Avg. Episode Reward | Avg. Wait (cars/step) | Improvement vs. LQF |
| :--- | :--- | :--- | :--- |
| **Trained DQN** | **-895.6** | **3.02** | **+17%** |
| Trained SARSA | -931.3 | 3.15 | +13% |
| Trained Q-Learning | -952.9 | 3.21 | +11% |
| Longest Queue First (LQF) | -1074.0 | 3.62 | Baseline |
| Fixed-Time (10-step) | -3127.6 | 10.44 | -191% |

### Key Findings

* **RL Surpasses Adaptive Rules:** Unlike LQF, which only reacts once long lines have already formed, the RL agents learned to anticipate traffic shifts, dropping the average waiting cars per step by 17%.
* **DQN Dominates:** The neural network achieved the best final score and reached optimal performance roughly 12.5 times faster than the tabular methods, demonstrating superior sample efficiency.
* **Reward Function Optimization:** An ablation study confirmed that a throughput-oriented reward function—which explicitly rewards successful vehicle departures while lightly penalizing wait times—yielded the most stable and effective learning compared to strict queue-based penalties.
* **Value Function Insights:** Visualizing the agent's learned strategy reveals that it correctly prioritizes the heavily trafficked main roads. It also effectively avoids extreme congestion states entirely, leaving those regions unvisited in its value map.

## Installation & Usage

Ensure you have Python 3.9+ installed along with Pygame, PyTorch, NumPy, and Matplotlib.

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```
**2. Run Training & Baseline Comparison:**
```Bash
python3 compare_algorithms.py
```
**3. Run the Reward Ablation Study:**
```Bash
python3 reward_ablation.py
```
**4. Visualize State-Value Functions:**
```Bash
python3 visualize_value_function.py
```
**5. Run Hyperparameter Grid Search:**
```Bash
python3 hyperparameter_tuning.py
```
**6. Launch Pygame Visualizers:** 
```Bash
# Run passive visualizer with the Trained DQN Agent
python3 visualizer.py --mode dqn

# Run passive visualizer with the LQF baseline
python3 visualizer.py --mode lqf

# Launch the Interactive Human vs. AI Split-Screen Game
python3 interactive_visualizer.py
```
