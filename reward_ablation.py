"""
Reward Function Ablation Study
================================
Trains a Q-Learning agent under 4 different reward function formulations and
evaluates each using a common, objective metric (average waiting cars/step).

Reward Modes:
  - "linear"     : R = -(ns + ew)               (default baseline)
  - "max_queue"  : R = -max(ns, ew)              (penalize worst bottleneck)
  - "quadratic"  : R = -(ns² + ew²) / 25        (heavier penalty for large queues)
  - "throughput" : R = departures_ns + departures_ew  (reward actual throughput)

Outputs:
  - reward_ablation_learning.png   : Learning curves for all 4 modes
  - reward_ablation_comparison.png : Bar charts of final evaluation metrics
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from environment import TrafficEnv
from agent import QLearningAgent

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Config ──────────────────────────────────────────────────────────────────
REWARD_MODES = {
    "Linear\n-(ns+ew)":         "linear",
    "Max-Queue\n-max(ns,ew)":   "max_queue",
    "Quadratic\n-(ns²+ew²)/25": "quadratic",
    "Throughput\n+departures":  "throughput",
}
TRAIN_EPISODES   = 800
EVAL_EPISODES    = 100
ALPHA            = 0.1
GAMMA            = 0.9
INITIAL_EPSILON  = 1.0
MIN_EPSILON      = 0.05
DECAY_RATE       = 0.995
MOVING_AVG_WIN   = 40

# Palette: one distinct colour per reward mode
PALETTE = ["#1e88e5", "#e53935", "#43a047", "#fb8c00"]


# ─── Training ────────────────────────────────────────────────────────────────
def train(env: TrafficEnv, agent: QLearningAgent) -> list:
    """Train a Q-Learning agent and return per-episode rewards."""
    epsilon = INITIAL_EPSILON
    episode_rewards = []

    for _ in range(TRAIN_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, ALPHA, GAMMA)
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    return episode_rewards


# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(agent: QLearningAgent) -> tuple:
    """
    Evaluate a trained agent using the NEUTRAL 'linear' environment so all
    reward modes are compared on the same objective scale.
    Returns (avg_episode_reward_linear, avg_waiting_cars_per_step).
    """
    eval_env = TrafficEnv(max_queue=5, max_steps=100, reward_mode="linear")
    total_rewards = []
    total_cars = 0
    total_steps = 0

    for _ in range(EVAL_EPISODES):
        state = eval_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            ns, ew, _ = state
            action = agent.choose_action(state, epsilon=0.0)
            state, reward, done, _ = eval_env.step(action)
            ep_reward += reward
            total_cars += (ns + ew)
            total_steps += 1

        total_rewards.append(ep_reward)

    return np.mean(total_rewards), total_cars / total_steps


# ─── Main ────────────────────────────────────────────────────────────────────
def run_ablation():
    all_rewards   = {}   # label -> list of episode rewards (training)
    eval_results  = {}   # label -> (avg_reward, avg_cars)

    for label, mode in REWARD_MODES.items():
        short = label.split("\n")[0]
        print(f"Training with reward_mode='{mode}' ({short})...")

        # Fresh environment and agent for each mode
        env   = TrafficEnv(max_queue=5, max_steps=100, reward_mode=mode)
        agent = QLearningAgent(max_queue=5)

        rewards = train(env, agent)
        all_rewards[label] = rewards

        avg_reward, avg_cars = evaluate(agent)
        eval_results[label] = (avg_reward, avg_cars)

        print(f"  → Eval avg reward: {avg_reward:.2f} | Avg waiting cars/step: {avg_cars:.3f}")

    _plot_learning_curves(all_rewards)
    _plot_comparison(eval_results)
    _print_table(eval_results)


# ─── Plotting ────────────────────────────────────────────────────────────────
def _moving_average(data: list, window: int) -> np.ndarray:
    return np.convolve(data, np.ones(window) / window, mode="valid")


def _plot_learning_curves(all_rewards: dict):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    for (label, rewards), color in zip(all_rewards.items(), PALETTE):
        ax.plot(rewards, alpha=0.12, color=color, linewidth=0.8)
        ma = _moving_average(rewards, MOVING_AVG_WIN)
        ax.plot(
            np.arange(MOVING_AVG_WIN - 1, len(rewards)),
            ma,
            color=color,
            linewidth=2.2,
            label=label.replace("\n", " — "),
        )

    ax.set_title(
        "Reward Function Ablation — Training Convergence",
        fontsize=14, fontweight="bold", color="white", pad=14
    )
    ax.set_xlabel("Episode", fontsize=11, color="#cccccc")
    ax.set_ylabel("Total Episode Reward (training scale)", fontsize=11, color="#cccccc")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    legend = ax.legend(
        fontsize=9, frameon=True, loc="lower right",
        facecolor="#1a1a2e", edgecolor="#555577", labelcolor="white"
    )
    ax.grid(True, linestyle="--", alpha=0.25, color="#555577")
    plt.tight_layout()
    plt.savefig("reward_ablation_learning.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print("\nSaved 'reward_ablation_learning.png'.")


def _plot_comparison(eval_results: dict):
    labels   = list(eval_results.keys())
    rewards  = [eval_results[l][0] for l in labels]
    avg_cars = [eval_results[l][1] for l in labels]

    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    short_labels = [l.replace("\n", "\n") for l in labels]

    for ax, values, title, ylabel, higher_better in [
        (ax1, rewards,  "Avg Episode Reward\n(Evaluated on Linear env — higher is better)",
         "Reward", True),
        (ax2, avg_cars, "Avg Waiting Cars / Step\n(Lower is better)",
         "Cars", False),
    ]:
        ax.set_facecolor("#0f0f1a")
        bars = ax.bar(
            range(len(labels)), values, color=PALETTE,
            edgecolor="#333355", linewidth=1.2, width=0.6
        )

        # Highlight the best bar
        best_idx = int(np.argmax(values) if higher_better else np.argmin(values))
        bars[best_idx].set_edgecolor("#ffd700")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(short_labels, fontsize=8.5, color="#dddddd")
        ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=10)
        ax.set_ylabel(ylabel, fontsize=10, color="#cccccc")
        ax.tick_params(axis="y", colors="#aaaaaa")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color="#555577")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        for i, (bar, val) in enumerate(zip(bars, values)):
            ypos = bar.get_height()
            offset = -0.5 if higher_better else 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos + offset,
                f"{val:.2f}",
                ha="center", va="bottom" if not higher_better else "top",
                fontsize=9, fontweight="bold",
                color="#ffd700" if i == best_idx else "#cccccc"
            )

    fig.suptitle(
        "Reward Function Ablation Study — Final Evaluation Comparison",
        fontsize=13, fontweight="bold", color="white", y=1.02
    )
    plt.savefig(
        "reward_ablation_comparison.png", dpi=200,
        facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.close()
    print("Saved 'reward_ablation_comparison.png'.")


def _print_table(eval_results: dict):
    print("\n" + "=" * 70)
    print(f"{'Reward Mode':<30} {'Avg Episode Reward':>20} {'Avg Cars/Step':>16}")
    print("=" * 70)
    for label, (avg_reward, avg_cars) in eval_results.items():
        short = label.replace("\n", " ")
        print(f"{short:<30} {avg_reward:>20.2f} {avg_cars:>16.3f}")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation()
