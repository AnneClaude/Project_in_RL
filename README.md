# Traffic Signal Control with Reinforcement Learning

This repository implements a suite of reinforcement learning agents (**Q-Learning**, **SARSA**, and **Deep Q-Networks (DQN)**) to optimize traffic signal switching at a 4-way intersection. By training on dynamic, time-varying traffic demand, the agents learn to coordinate green light cycles, adapt to rush hours, and minimize vehicle waiting times compared to traditional heuristics.

---

## 🚦 The Environment

The intersection is implemented as a discrete simulation (`environment.py`) with queue caps and realistic transition constraints:
* **State Space**: A tuple of `(ns_queue, ew_queue, green_light_state)` where:
  * `ns_queue` and `ew_queue` are queue lengths capped at `5`.
  * `green_light_state` indicates which lane currently has the green phase:
    * `0`: North/South Green
    * `1`: East/West Green
    * `2`: North/South Yellow (transitioning to EW Green)
    * `3`: East/West Yellow (transitioning to NS Green)
* **Action Space**: Binary choice at each step:
  * `0`: Keep current light state.
  * `1`: Switch phase (initiates a 1-step yellow light transition).
* **Constraints**:
  * **Minimum Green Duration**: Green signals must be held for at least `3` steps before a switch is allowed, preventing erratic toggling.
  * **Yellow Transition Penalty**: Switching lights triggers a `1`-step yellow light phase during which all departures are blocked (departures = 0). This creates a transition delay cost that the agent must optimize.
* **Dynamic Traffic Demand**: The simulation models a full day divided into three distinct flow patterns over 100 steps:
  * **Morning Rush (Steps 0–30)**: Heavy North/South flow (80% spawn chance), light East/West flow (10% spawn chance).
  * **Mid-day Off-Peak (Steps 31–70)**: Balanced traffic (30% spawn chance for both directions).
  * **Evening Rush (Steps 71–100)**: Light North/South flow (10% spawn chance), heavy East/West flow (80% spawn chance).

---

## 🧠 Core Algorithms

### 1. Baselines
* **Random Switch**: Uniformly random selection between keeping the light or switching.
* **Fixed-Time Switch (5 & 10 steps)**: Switches signals at rigid, periodic step intervals.
* **Longest Queue First (LQF)**: A dynamic heuristic that switches the light to green for the lane with the longer queue, subject to the 3-step minimum green constraint.

### 2. Tabular Reinforcement Learning
* **Q-Learning**: Off-policy temporal-difference (TD) control. Learns the optimal state-value function independent of the agent's current action selection policy.
* **SARSA**: On-policy TD control. Updates the action-value function based on the actual actions selected under the exploration policy ($\epsilon$-greedy), leading to safer policy convergence under transition penalties.

### 3. Deep Reinforcement Learning
* **Deep Q-Networks (DQN)**: Implemented using **PyTorch** (`dqn_agent.py`). Uses a neural network function approximator to predict Q-values.
  * State inputs are normalized and green light phases are represented using one-hot encoding (continuous state vector of size 6).
  * Implements **Experience Replay** to break temporal correlation and a **Target Network** to stabilize target updates.

---

## 📊 Performance Comparison

After training Q-Learning, SARSA, and DQN for `1,000` episodes, we evaluated each policy over `100` test episodes under dynamic demand:

| Strategy | Avg Episode Reward | Avg Waiting Cars/Step | Performance vs. Random |
| :--- | :---: | :---: | :---: |
| **Trained DQN** | **-229.26** | **2.31** | **+32.5%** |
| **Trained Q-Learning** | **-236.92** | **2.39** | **+30.2%** |
| **Trained SARSA** | **-242.70** | **2.45** | **+28.5%** |
| Longest Queue First (LQF) | -246.62 | 2.49 | +27.4% |
| Random Switch | -339.49 | 3.40 | Baseline |
| Fixed-Time Switch (5-steps) | -343.89 | 3.46 | -1.3% |
| Fixed-Time Switch (10-steps) | -352.56 | 3.54 | -3.8% |

### Key Findings
1. **DQN Wins**: DQN achieves the lowest queue accumulation (2.31 cars/step) because its normalized vector state representation and continuous weight optimization allow it to capture finer state-action nuances compared to tabular representations.
2. **RL Outperforms LQF**: While LQF is a strong dynamic heuristic, it operates greedily. The RL agents learn to anticipate the cost of the yellow transition phase and time switches strategically, yielding significantly higher total rewards.
3. **Fixed-Time Fails**: Fixed-time timing plans perform poorly because they cannot adapt to asymmetric rush-hour flows, causing major vehicle backups.

*Plots of the learning curves and final evaluations are saved automatically as `learning_curves.png` and `policy_comparison.png`.*

---

## 🧪 Reward Function Ablation Study

To understand the impact of reward design, we evaluated four different reward formulations using the Q-Learning agent. To guarantee scientific reliability, each configuration was trained across **5 independent random seeds** and evaluated over **500 episodes**:

1. **Linear Queue Penalty** ($R = -(ns + ew)$): Baseline penalty. Treats all waiting queues linearly.
2. **Max-Queue Penalty** ($R = -\max(ns, ew)$): penalizes only the worst-performing bottleneck lane.
3. **Quadratic Queue Penalty** ($R = -(ns^2 + ew^2)$): Penalizes larger queues progressively, discouraging high congestion.
4. **Throughput-Oriented** ($R = departures - 0.1 \times (ns + ew)$): Directly rewards successfully cleared vehicles (throughput) while including a minor queue penalty.

### Ablation Results (Mean ± Std over 5 Seeds)

| Reward Formulation | Mean Avg Waiting Cars/Step | Standard Deviation (±) | Seed Consistency (Win/Loss) |
| :--- | :---: | :---: | :---: |
| **Throughput-Oriented** | **2.336** | **0.023** | **5/5 (Clear Winner)** |
| **Linear Queue Penalty** | **2.438** | **0.058** | 4/5 vs. Quadratic |
| **Quadratic Queue Penalty** | **2.463** | **0.032** | 1/5 vs. Linear |
| **Max-Queue Penalty** | **2.545** | **0.061** | 0/5 (Worst) |

### Key Takeaways
* **Maximize Flow directly**: Directly rewarding vehicle departures aligns the reinforcement learning objective with the objective evaluation metric (clearing the intersection). It achieves the overall lowest waiting times and highest stability.
* **Bottlenecks and Gradients**: The **Max-Queue** penalty is the least effective because it does not penalize queue growth on the secondary lane, leaving the agent without a learning signal for that lane until it becomes the absolute worst bottleneck.

*Ablation learning curves and bar graphs are saved as `reward_ablation_learning.png` and `reward_ablation_comparison.png`.*

---

## 📈 State-Value & Decision Boundaries

Running `visualize_value_function.py` generates heatmaps (`value_functions_comparison.png`) of the learned state-value function $V(s) = \max_a Q(s, a)$ and decision boundaries for keeping vs. switching lights:

* **Queue Accumulation cost**: States turn deep red (highly negative value) as both queue sizes approach the maximum limit of 5.
* **Phase Asymmetry**: Since the North/South direction has a heavy morning rush probability (80% spawn rate), the agent learns an asymmetric decision boundary. When NS is green, the agent holds the signal green even if the East/West queue builds up to 2 or 3. However, when EW is green, the agent switches back to NS green as soon as the NS queue reaches 1 or 2, reflecting its adaptation to the asymmetric traffic distribution.

---

## 🎮 Pygame Visualizers

The project features two Pygame-based GUI visualizers:
1. **Passive Agent Visualizer** (`visualizer.py`): Renders a real-time 2D simulation of the intersection, letting you watch the trained Q-learning/SARSA agents or LQF/Fixed-time heuristics control traffic.
2. **Interactive Human vs. AI Game** (`interactive_visualizer.py`): A split-screen challenge! You control the left intersection manually while the trained SARSA agent controls the right intersection. Play through rush hours and see if you can beat the AI.

**Controls**:
* `SPACEBAR`: Switch signals (triggers yellow transition phase).
* `R`: Reset the episode.

---

## ⚙️ Installation & Usage

### 1. Requirements
Ensure Python 3.9+ and Pygame, PyTorch, NumPy, and Matplotlib are installed:
```bash
pip install -r requirements.txt
```

### 2. Running Training & Baseline Comparison
Trains Q-Learning, SARSA, and DQN for 1000 episodes and plots performance:
```bash
python3 compare_algorithms.py
```

### 3. Running the Reward Ablation Study
Runs the 5-seed, 500-evaluation-episode ablation study across the four reward modes:
```bash
python3 reward_ablation.py
```

### 4. Visualizing State-Value Functions
Generates the $V(s)$ value function heatmaps and decision boundary maps:
```bash
python3 visualize_value_function.py
```

### 5. Running Hyperparameter Grid Search
Tunes tabular hyperparameter rates ($\alpha$, $\gamma$) and outputs a sensitivity heatmap:
```bash
python3 hyperparameter_tuning.py
```

### 6. Launching Pygame Visualizers
```bash
# Run passive visualizer with Trained Q-Learning Agent
python3 visualizer.py --mode q_learning

# Run passive visualizer with LQF heuristic
python3 visualizer.py --mode lqf

# Launch the Human vs. AI Split-Screen Game
python3 interactive_visualizer.py
```
