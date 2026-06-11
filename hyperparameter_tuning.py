import numpy as np
import random
import matplotlib.pyplot as plt
from environment import TrafficEnv
from agent import QLearningAgent
from compare_algorithms import train_q_learning, evaluate_policy

def run_tuning():
    random.seed(42)
    np.random.seed(42)
    
    env = TrafficEnv(max_queue=5, max_steps=100)
    
    # Parameter grid
    alphas = [0.01, 0.05, 0.1, 0.2]
    gammas = [0.8, 0.9, 0.99]
    
    grid_rewards = np.zeros((len(alphas), len(gammas)))
    grid_cars = np.zeros((len(alphas), len(gammas)))
    
    print("Starting hyperparameter grid search (Q-learning, 500 training / 50 eval episodes per run)...")
    print("-" * 70)
    print(f"{'Alpha':<10}{'Gamma':<10}{'Avg Reward':<15}{'Avg Waiting Cars'}")
    print("-" * 70)
    
    for i, alpha in enumerate(alphas):
        for j, gamma in enumerate(gammas):
            # Train agent
            agent = QLearningAgent(max_queue=5)
            train_q_learning(env, agent, num_episodes=500, alpha=alpha, gamma=gamma, initial_epsilon=1.0)
            
            # Evaluate agent
            avg_reward, avg_cars = evaluate_policy(env, agent, mode="greedy", num_episodes=50)
            grid_rewards[i, j] = avg_reward
            grid_cars[i, j] = avg_cars
            
            print(f"{alpha:<10}{gamma:<10}{avg_reward:<15.2f}{avg_cars:<15.2f}")
            
    print("-" * 70)
    print("\nTuning complete! Generating hyperparameter tuning heatmap plot...")
    
    # Plotting Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    # Using 'YlGn' colormap where darker green represents better (closer to 0) values
    im = ax.imshow(grid_rewards, cmap="YlGn", aspect="auto")
    
    # Configure axes
    ax.set_xticks(np.arange(len(gammas)))
    ax.set_yticks(np.arange(len(alphas)))
    ax.set_xticklabels(gammas)
    ax.set_yticklabels(alphas)
    
    # Center text labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Determine text colors for readability based on color intensity
    min_val, max_val = grid_rewards.min(), grid_rewards.max()
    val_range = max_val - min_val if max_val != min_val else 1.0
    
    # Annotate cell values
    for i in range(len(alphas)):
        for j in range(len(gammas)):
            val = grid_rewards[i, j]
            # Use light text for darker backgrounds, and vice versa
            normalized_val = (val - min_val) / val_range
            text_color = "white" if normalized_val > 0.5 else "black"
            
            ax.text(j, i, f"{val:.1f}\n({grid_cars[i, j]:.2f} cars)",
                    ha="center", va="center", color=text_color, fontweight="bold")
            
    ax.set_title("Hyperparameter Grid Search (Q-Learning)\nAverage Episode Reward & Average Waiting Cars per Step", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Discount Factor (gamma)", fontsize=11, labelpad=10)
    ax.set_ylabel("Learning Rate (alpha)", fontsize=11, labelpad=10)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Average Episode Reward (Higher is Better)", fontsize=10, labelpad=10)
    
    plt.tight_layout()
    plt.savefig("hyperparameter_tuning.png", dpi=300)
    plt.close()
    print("Saved 'hyperparameter_tuning.png'.")

if __name__ == "__main__":
    run_tuning()
