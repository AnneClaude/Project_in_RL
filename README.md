# Traffic Signal Control with Reinforcement Learning

This project implements reinforcement learning agents (**Q-Learning** and **SARSA**) to optimize traffic signal switching at a 4-way intersection under realistic conditions.

## The Environment
The intersection is modeled as a discrete simulation with:
* **North/South Lane:** Heavy traffic flow (60% vehicle spawn chance per step).
* **East/West Lane:** Light traffic flow (20% vehicle spawn chance per step).
* **Constraints**:
  * **3-step minimum green light duration** to prevent rapid toggling.
  * **1-step Yellow Light Transition Phase** during which all vehicle departures are blocked (departures = 0), penalizing frequent switching.
* **Reward**: The negative sum of waiting cars in both queues.

## Evaluation Results

After training the Q-Learning and SARSA agents for 1,000 episodes, we evaluated their performance against three baseline controllers over 100 test episodes:

### Policy Comparison (Averaged over 100 episodes)

| Strategy | Avg Episode Reward | Avg Waiting Cars/Step |
| :--- | :--- | :--- |
| **Trained SARSA** | **-266.67** | **2.69** |
| **Trained Q-Learning** | **-266.81** | **2.69** |
| Longest Queue First (LQF) | -277.19 | 2.79 |
| Fixed-Time Switch (5-steps) | -354.47 | 3.56 |
| Random Switch | -364.28 | 3.66 |
| Fixed-Time Switch (10-steps) | -380.23 | 3.81 |

## Key Findings & Conclusion
* **RL Beats Greedy Heuristics**: The Longest Queue First (LQF) heuristic is a strong greedy controller. However, because switching signals now incurs a yellow light delay penalty (blocking traffic flow), LQF switches too frequently and loses efficiency.
* **Smart Lookahead**: The RL agents (Q-Learning and SARSA) successfully learned this transition cost, choosing to hold green lights longer in the busier N/S direction. As a result, they outperform LQF and reduce average wait times by **~25-30%** compared to traditional fixed-time cycles.

## How to Run

### 1. Train and Evaluate
To retrain the agents and generate performance comparison plots (`learning_curves.png`, `policy_comparison.png`):
```bash
python3 compare_algorithms.py
```

### 2. Run Hyperparameter Tuning
To perform a grid search on learning rate ($\alpha$) and discount factor ($\gamma$) and generate a heatmap (`hyperparameter_tuning.png`):
```bash
python3 hyperparameter_tuning.py
```

### 3. Interactive Pygame Visualizer
Run the visualizer with your desired control agent or heuristic:
```bash
# Run using the trained SARSA agent
python3 visualizer.py --mode sarsa

# Run using the trained Q-learning agent
python3 visualizer.py --mode q_learning

# Run using the Longest Queue First (LQF) heuristic
python3 visualizer.py --mode lqf

# Run using a 5-step Fixed-Time cycle
python3 visualizer.py --mode fixed
```
