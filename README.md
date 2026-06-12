# Traffic Signal Control with Reinforcement Learning

This project implements reinforcement learning agents (**Q-Learning** and **SARSA**) to optimize traffic signal switching at a 4-way intersection under realistic conditions.

## The Environment
The intersection is modeled as a discrete simulation with time-varying traffic demand across three distinct daily phases:
* **Morning Rush (Steps 0–30)**: Heavy North/South flow (80% spawn chance) and light East/West flow (10% spawn chance).
* **Mid-day Off-Peak (Steps 31–70)**: Balanced flow (30% spawn chance on both N/S and E/W).
* **Evening Rush (Steps 71–100)**: Light North/South flow (10% spawn chance) and heavy East/West flow (80% spawn chance).
* **Constraints**:
  * **3-step minimum green light duration** to prevent rapid toggling.
  * **1-step Yellow Light Transition Phase** during which all vehicle departures are blocked (departures = 0), penalizing frequent switching.
* **Reward**: The negative sum of waiting cars in both queues.

## Evaluation Results

After training the Q-Learning and SARSA agents for 1,000 episodes, we evaluated their performance against three baseline controllers over 100 test episodes under dynamic traffic demand:

### Policy Comparison (Averaged over 100 episodes)

| Strategy | Avg Episode Reward | Avg Waiting Cars/Step |
| :--- | :--- | :--- |
| **Trained SARSA** | **-239.55** | **2.42** |
| **Trained Q-Learning** | **-242.30** | **2.45** |
| Longest Queue First (LQF) | -248.42 | 2.51 |
| Fixed-Time Switch (5-steps) | -336.73 | 3.38 |
| Random Switch | -342.11 | 3.43 |
| Fixed-Time Switch (10-steps) | -352.94 | 3.54 |

## Key Findings & Conclusion
* **RL Beats Greedy Heuristics**: Under dynamic, time-varying traffic demand, the RL agents successfully adapted to changing flow distributions, allocating green time to the busiest direction on-the-fly. They outperform the Longest Queue First (LQF) heuristic by optimizing switching choices around the yellow transition cost.
* **Robust Adaptation**: Without knowing the explicit time or phase, the RL agents rely purely on queue states to hold lights longer in active directions, reducing average waiting cars by **~30%** compared to standard fixed-time controllers.

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
