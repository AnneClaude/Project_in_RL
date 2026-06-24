"""
Reward Function Ablation Study
================================
Trains a Q-Learning agent under 4 different reward function formulations and
evaluates each using a common, objective metric (average waiting cars/step).
Each mode is run over 5 independent random seeds; results are reported as
mean ± standard deviation for statistical reliability.

Reward Modes:
  - "linear"     : R = -(ns + ew)                         (default baseline)
  - "max_queue"  : R = -max(ns, ew)                        (penalize worst bottleneck)
  - "quadratic"  : R = -(ns² + ew²)                        (strong penalty for large queues)
  - "throughput" : R = (departures) - 0.1*(ns+ew)          (reward throughput, lightly penalize queues)

Outputs:
  - reward_ablation_learning.png   : Learning curves (mean ± std band) for all 4 modes
  - reward_ablation_comparison.png : Bar charts with error bars for final evaluation metrics
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from environment import TrafficEnv
from agent import QLearningAgent

# ─── Config ──────────────────────────────────────────────────────────────────
REWARD_MODES = {
    "Linear\n-(ns+ew)":             "linear",
    "Max-Queue\n-max(ns,ew)":       "max_queue",
    "Quadratic\n-(ns²+ew²)":        "quadratic",
    "Throughput\ndep-0.1*(ns+ew)":  "throughput",
}
SEEDS            = [42, 123, 777, 999, 2024]   # 5 independent seeds for statistical validity
TRAIN_EPISODES   = 1200
EVAL_EPISODES    = 500
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
    # label -> list of (avg_reward, avg_cars) tuples across seeds
    seed_eval:    dict[str, list] = {label: [] for label in REWARD_MODES}
    # label -> (n_episodes, n_seeds) reward matrix for learning curves
    seed_rewards: dict[str, list] = {label: [] for label in REWARD_MODES}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        random.seed(seed)
        np.random.seed(seed)

        for label, mode in REWARD_MODES.items():
            short = label.split("\n")[0]
            print(f"  Training reward_mode='{mode}' ({short})...", end=" ", flush=True)

            env   = TrafficEnv(max_queue=5, max_steps=100, reward_mode=mode)
            agent = QLearningAgent(max_queue=5)

            rewards = train(env, agent)
            seed_rewards[label].append(rewards)

            avg_reward, avg_cars = evaluate(agent)
            seed_eval[label].append((avg_reward, avg_cars))

            print(f"reward={avg_reward:.2f}  cars/step={avg_cars:.3f}")

    # Aggregate across seeds
    eval_results = {}   # label -> (mean_reward, std_reward, mean_cars, std_cars)
    for label in REWARD_MODES:
        rewards_list = [r for r, _ in seed_eval[label]]
        cars_list    = [c for _, c in seed_eval[label]]
        eval_results[label] = (
            np.mean(rewards_list), np.std(rewards_list),
            np.mean(cars_list),    np.std(cars_list),
        )

    _plot_learning_curves(seed_rewards)
    _plot_comparison(eval_results)
    _print_table(eval_results)


# ─── Plotting ────────────────────────────────────────────────────────────────
def _moving_average(data: list, window: int) -> np.ndarray:
    return np.convolve(data, np.ones(window) / window, mode="valid")


def _plot_learning_curves(seed_rewards: dict):
    """Plot mean learning curve ± std band across seeds for each reward mode."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    for (label, runs), color in zip(seed_rewards.items(), PALETTE):
        # Stack into (n_seeds, n_episodes) and smooth each run
        smoothed = np.array([_moving_average(r, MOVING_AVG_WIN) for r in runs])
        mean_curve = smoothed.mean(axis=0)
        std_curve  = smoothed.std(axis=0)
        x = np.arange(MOVING_AVG_WIN - 1, TRAIN_EPISODES)

        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.18, color=color)
        ax.plot(x, mean_curve, color=color, linewidth=2.2,
                label=label.replace("\n", " — "))

    ax.set_title(
        "Reward Function Ablation — Training Convergence (mean ± std, 5 seeds)",
        fontsize=13, fontweight="bold", color="white", pad=14
    )
    ax.set_xlabel("Episode", fontsize=11, color="#cccccc")
    ax.set_ylabel("Total Episode Reward (training scale)", fontsize=11, color="#cccccc")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    ax.legend(fontsize=9, frameon=True, loc="lower right",
              facecolor="#1a1a2e", edgecolor="#555577", labelcolor="white")
    ax.grid(True, linestyle="--", alpha=0.25, color="#555577")
    plt.tight_layout()
    plt.savefig("reward_ablation_learning.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print("\nSaved 'reward_ablation_learning.png'.")


def _plot_comparison(eval_results: dict):
    """Bar charts with error bars for reward and waiting cars."""
    labels     = list(eval_results.keys())
    mean_r     = [eval_results[l][0] for l in labels]
    std_r      = [eval_results[l][1] for l in labels]
    mean_cars  = [eval_results[l][2] for l in labels]
    std_cars   = [eval_results[l][3] for l in labels]

    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax, means, stds, title, ylabel, higher_better in [
        (ax1, mean_r,    std_r,
         "Avg Episode Reward\n(Evaluated on Linear env — higher is better)", "Reward", True),
        (ax2, mean_cars, std_cars,
         "Avg Waiting Cars / Step\n(Lower is better)", "Cars", False),
    ]:
        ax.set_facecolor("#0f0f1a")
        bars = ax.bar(range(len(labels)), means, color=PALETTE,
                      edgecolor="#333355", linewidth=1.2, width=0.6)
        ax.errorbar(range(len(labels)), means, yerr=stds,
                    fmt="none", color="white", capsize=5, linewidth=1.5, capthick=1.5)

        # Highlight best bar
        best_idx = int(np.argmax(means) if higher_better else np.argmin(means))
        bars[best_idx].set_edgecolor("#ffd700")
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5, color="#dddddd")
        ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=10)
        ax.set_ylabel(ylabel, fontsize=10, color="#cccccc")
        ax.tick_params(axis="y", colors="#aaaaaa")
        ax.grid(True, axis="y", linestyle="--", alpha=0.25, color="#555577")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        for i, (bar, val, std) in enumerate(zip(bars, means, stds)):
            ypos   = bar.get_height()
            offset = -(abs(std) + 1.5) if higher_better else (abs(std) + 0.02)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos + offset,
                    f"{val:.2f}\n±{std:.2f}",
                    ha="center", va="bottom" if not higher_better else "top",
                    fontsize=8, fontweight="bold",
                    color="#ffd700" if i == best_idx else "#cccccc")

    fig.suptitle(
        "Reward Function Ablation Study — Final Evaluation (mean ± std, 5 seeds)",
        fontsize=13, fontweight="bold", color="white", y=1.02
    )
    plt.savefig("reward_ablation_comparison.png", dpi=200,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print("Saved 'reward_ablation_comparison.png'.")


def _print_table(eval_results: dict):
    print("\n" + "=" * 80)
    print(f"{'Reward Mode':<30} {'Mean Reward':>14} {'±Std':>8} {'Mean Cars/Step':>16} {'±Std':>8}")
    print("=" * 80)
    for label, (mr, sr, mc, sc) in eval_results.items():
        short = label.replace("\n", " ")
        print(f"{short:<30} {mr:>14.2f} {sr:>8.2f} {mc:>16.3f} {sc:>8.3f}")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation()
