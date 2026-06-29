# Traffic Signal Control using Reinforcement Learning

**Final Project — Introduction to Reinforcement Learning**  
**Silin Michael · Anne Claude Abinader**  
**Instructor: Dr. Teddy Lazebnik**

---

This repository implements three model-free reinforcement learning algorithms — **Q-Learning**, **SARSA**, and **Deep Q-Network (DQN)** — to optimize traffic signal switching at a simulated four-way intersection. By framing the problem as a Markov Decision Process (MDP), the agents learn adaptive control policies that significantly outperform traditional rule-based controllers.

---

## The Environment (`environment.py`)

A custom 4-way intersection simulation with **dynamic, phase-based traffic demand**:

| Episode Steps | Phase | NS Arrival Rate | EW Arrival Rate |
|---|---|---|---|
| 0 – 29 | Morning Rush | 80% | 10% |
| 30 – 69 | Mid-day Balanced | 30% | 30% |
| 70 – 100 | Evening Rush | 10% | 80% |

- **State Space:** `(ns_queue, ew_queue, light_phase)` — queue lengths capped at **15** during full training (1,024 possible states), capped at **5** in real-time visualizer demos for display clarity.
- **Light Phases:** 4 discrete states — NS Green (0), EW Green (1), NS Yellow (2), EW Yellow (3). Yellow phases model realistic transition delays.
- **Action Space:** Binary — `0` = keep current light, `1` = initiate switch. A **3-step minimum green time** constraint prevents rapid, inefficient toggling.
- **Reward Function (primary):** `−(ns_queue + ew_queue)` per step — penalizes the total number of vehicles waiting. Four modes are available for ablation: `linear`, `quadratic`, `max_queue`, `throughput`.

---

## Algorithms & Baselines

### RL Agents (`agent.py`, `dqn_agent.py`)

| Agent | Type | Update Rule |
|---|---|---|
| **Q-Learning** | Off-policy TD | `Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',·) − Q(s,a)]` |
| **SARSA** | On-policy TD | `Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]` |
| **DQN** | Deep RL | Neural net (6→64→64→2) + Experience Replay + Target Network |

DQN uses a normalized 6-dimensional state vector: `[ns/max_q, ew/max_q, one-hot(phase)]`, trained on Apple MPS (Metal) acceleration.

### Non-Learning Baselines (`baseline.py`, `compare_algorithms.py`)
- **Fixed-Time (5-step):** Switch every 5 steps regardless of traffic.
- **Fixed-Time (10-step):** Switch every 10 steps.
- **Longest Queue First (LQF):** Greedy rule — switch when the idle direction has more cars than the active direction and minimum green time has elapsed.
- **Random Switch:** Uniform random action at each step.

---

## Results

All agents trained with 3 random seeds and evaluated over **200 greedy episodes**. Full training: 20,000 episodes (tabular) / 2,500 episodes (DQN).

| Strategy | Avg. Episode Reward | Avg. Wait (cars/step) | vs. Best Heuristic (LQF) |
|:---|:---:|:---:|:---:|
|  **Trained DQN** | **−895.6** | **3.02** | **+17%** |
|  Trained SARSA | −931.3 | 3.15 | +13% |
|  Trained Q-Learning | −952.9 | 3.21 | +11% |
| Longest Queue First (LQF) | −1074.0 | 3.62 | — |
| Fixed-Time (10-step) | −3127.6 | 10.44 | −191% |
| Random Switch | −3381.6 | 11.28 | −215% |
| Fixed-Time (5-step) | −3511.3 | 11.71 | −227% |

### Key Findings

- **All RL agents dramatically outperform all baselines** — a 17% improvement over the best heuristic (LQF) and a ~73% improvement over fixed-time control.
- **DQN achieves the best final score**, with the neural network generalizing across queue states more effectively than tabular methods.
- **Fixed-Time (5-step) is worse than random** — switching too frequently causes constant yellow phases where no vehicles can depart, demonstrating the necessity of learned timing.
- **LQF reacts; RL anticipates.** LQF only switches after a queue imbalance has already formed. RL agents learn to preemptively adapt as dynamic demand phases shift throughout the episode.
- **Reward ablation study** confirmed that a throughput-oriented reward function (`departures − 0.1×queues`) yields the most stable learning across seeds.

---

## Repository Structure

```
Project_in_RL/
├── environment.py              # Traffic intersection MDP
├── agent.py                    # Q-Learning & SARSA tabular agents
├── dqn_agent.py                # DQN agent (PyTorch, Experience Replay, Target Net)
├── main.py                     # Simple single-run Q-Learning trainer
├── compare_algorithms.py       # Train all 3 agents + baselines + comparison plots
├── train_full.py               # Full-scale training (3 seeds, 20k/2.5k episodes)
├── baseline.py                 # Standalone fixed-time baseline runner
├── reward_ablation.py          # Ablation study across 4 reward formulations
├── hyperparameter_tuning.py    # Grid search over α, γ, ε-decay
├── visualize_value_function.py # 3D/2D value function heatmaps
├── visualizer.py               # Live Pygame dashboard (passive, all modes)
├── interactive_visualizer.py   # Human vs. AI interactive game (SPACE to switch)
├── q_table_qlearning.npy       # Trained Q-Learning table (max_queue=5, 1k ep)
├── q_table_sarsa.npy           # Trained SARSA table (max_queue=5, 1k ep)
├── dqn_model.pth               # Trained DQN weights (max_queue=5, 1k ep)
├── full_training_results/      # Full-scale models, logs, and evaluation plots
│   ├── eval_results.json       # Final greedy evaluation numbers (all 7 strategies)
│   ├── full_learning_curves.png
│   ├── full_policy_comparison.png
│   ├── full_convergence_detail.png
│   ├── dqn_model_seed{1,2,3}_final.pth
│   ├── q_table_qlearning_seed{1,2,3}_final.npy
│   └── q_table_sarsa_seed{1,2,3}_final.npy
├── result_images/              # Saved plots from all experiments
├── requirements.txt
└── README.md
```

---

## Installation & Usage

**Requirements:** Python 3.9+, PyTorch, NumPy, Matplotlib, Pygame.

```bash
pip install -r requirements.txt
```

### Quick comparison (trains from scratch, ~2 min)
```bash
python3 compare_algorithms.py
```

### Full-scale production training (3 seeds, ~90 min)
```bash
python3 train_full.py
```

### Reward function ablation study
```bash
python3 reward_ablation.py
```

### Hyperparameter grid search
```bash
python3 hyperparameter_tuning.py
```

### State-value function visualisation
```bash
python3 visualize_value_function.py
```

### Live agent dashboard (passive visualizer)
```bash
# Watch any trained agent with a live reward-history chart
python3 visualizer.py --mode q_learning   # Trained Q-Learning
python3 visualizer.py --mode sarsa        # Trained SARSA
python3 visualizer.py --mode dqn          # Trained DQN
python3 visualizer.py --mode lqf          # Longest Queue First heuristic
python3 visualizer.py --mode fixed        # Fixed-Time baseline
python3 visualizer.py --mode random       # Random Switch
```

### Interactive Human vs. AI game
```bash
python3 interactive_visualizer.py
# Press SPACE to switch the traffic light
# Press R   to restart the episode
```

---

## Unique Contributions

1. **4-phase traffic light model** with realistic yellow transitions (most implementations use 2 phases).
2. **3-phase dynamic demand** (morning / midday / evening rush) within a single episode.
3. **Live reward-history chart** in the visualizer — a scrolling line graph showing per-step rewards in real time.
4. **Human vs. AI interactive mode** — play against the trained agent on identical traffic conditions.
5. **Reward function ablation** across 4 formulations with reproducible results.
6. **DQN normalization bug diagnosed and fixed** — raw queue values fed to the neural network caused gradient instability; normalizing to `[0,1]` restored DQN's #1 ranking.