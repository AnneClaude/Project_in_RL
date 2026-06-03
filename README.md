# Traffic Signal Control with Reinforcement Learning

This project implements a simple Q-Learning agent to optimize traffic signal switching at a 4-way intersection.

## The Environment
The intersection has two main queues:
* **North/South Lane:** Heavy traffic flow (60% spawn chance per step)
* **East/West Lane:** Light traffic flow (20% spawn chance per step)

The environment has a **3-step minimum green light constraint**, meaning the light cannot be switched back-to-back rapidly. The agent receives a negative reward equal to the total number of waiting cars in both queues, encouraging it to clear the traffic as efficiently as possible.

## Evaluation Results

After training the Q-Learning agent for 1,000 episodes, we evaluated its performance against two non-RL algorithms over 100 test episodes:
1. **Trained Q-Learning:** Chooses the optimal action based on the learned Q-table.
2. **Fixed-Time Switch:** A naive approach that cycles the light every 5 steps (or every 10 steps in the baseline script).
3. **Random Switch:** Chooses to keep or switch the light randomly.

### Policy Comparison (Averaged over 100 episodes)

| Strategy | Avg Episode Reward | Avg Waiting Cars/Step |
| :--- | :--- | :--- |
| **Trained Q-Learning** | **-197.09** | **2.00** |
| Fixed-Time Switch (5-steps) | -279.00 | 2.81 |
| Random Switch | -284.79 | 2.87 |
| Fixed-Time Switch (10-steps) | -353.06 | 3.54 |

*Note: The Fixed-Time (10-steps) result was tested separately using `baseline.py` over 50 episodes.*

## Conclusion
The RL agent successfully learns to adapt to the asymmetric traffic conditions. Because the North/South lane has 3x more traffic, the agent learns to keep the North/South light green significantly longer than the East/West light, leading to a massive ~30-40% reduction in average wait times compared to rigid, traditional algorithms.

## How to Run
* `python main.py` to train the agent, run the evaluation, and generate the `learning_curve.png` plot.
* `python baseline.py` to test the rigid 10-step fixed-timer algorithm.
