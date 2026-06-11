
import numpy as np
import random
import matplotlib.pyplot as plt
from environment import TrafficEnv
from agent import QLearningAgent, SARSAAgent

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def train_q_learning(
    env: TrafficEnv, 
    agent: QLearningAgent, 
    num_episodes: int = 1000, 
    alpha: float = 0.1, 
    gamma: float = 0.9, 
    initial_epsilon: float = 1.0, 
    min_epsilon: float = 0.05, 
    decay_rate: float = 0.995
):
    """Trains a Q-Learning agent."""
    epsilon = initial_epsilon
    episode_rewards = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, alpha, gamma)
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)
        
    return episode_rewards

def train_sarsa(
    env: TrafficEnv, 
    agent: SARSAAgent, 
    num_episodes: int = 1000, 
    alpha: float = 0.1, 
    gamma: float = 0.9, 
    initial_epsilon: float = 1.0, 
    min_epsilon: float = 0.05, 
    decay_rate: float = 0.995
):
    """Trains a SARSA agent."""
    epsilon = initial_epsilon
    episode_rewards = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        # In SARSA, we choose the first action before entering the loop
        action = agent.choose_action(state, epsilon)
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            
            # Choose next action on-policy
            next_action = agent.choose_action(next_state, epsilon)
            
            # Update using chosen next action
            agent.update_q_table(state, action, reward, next_state, next_action, alpha, gamma)
            
            state = next_state
            action = next_action
            total_reward += reward
            
        episode_rewards.append(total_reward)
        epsilon = max(min_epsilon, epsilon * decay_rate)
        
    return episode_rewards

def evaluate_policy(env: TrafficEnv, agent=None, mode: str = "greedy", num_episodes: int = 100, switch_interval: int = 5):
    """
    Evaluates a policy and returns average reward and waiting cars per step.
    
    Modes:
      - "greedy": RL agent with epsilon=0
      - "random": random action at each step
      - "fixed": switch light every `switch_interval` steps
      - "lqf": Longest Queue First heuristic
    """
    total_rewards = []
    total_waiting_cars = 0
    total_steps = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            ns_queue, ew_queue, green_light = state
            
            if mode == "greedy" and agent is not None:
                action = agent.choose_action(state, epsilon=0.0)
            elif mode == "fixed":
                action = 1 if step_count > 0 and step_count % switch_interval == 0 else 0
            elif mode == "lqf":
                # LQF / MaxPressure rule: switch if current light queue is shorter than the other,
                # and minimum green time (3 steps) has passed.
                if green_light == 0:  # NS has green
                    if ew_queue > ns_queue and env.time_since_switch >= 3:
                        action = 1
                    else:
                        action = 0
                else:  # EW has green
                    if ns_queue > ew_queue and env.time_since_switch >= 3:
                        action = 1
                    else:
                        action = 0
            else:  # random
                action = random.choice([0, 1])
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_waiting_cars += (ns_queue + ew_queue)
            total_steps += 1
            
            state = next_state
            step_count += 1
            
        total_rewards.append(episode_reward)
        
    avg_reward = np.mean(total_rewards)
    avg_waiting_cars = total_waiting_cars / total_steps
    return avg_reward, avg_waiting_cars

def run_comparison():
    env = TrafficEnv(max_queue=5, max_steps=100)
    
    # Initialize agents
    q_agent = QLearningAgent(max_queue=5)
    sarsa_agent = SARSAAgent(max_queue=5)
    
    # Train agents
    q_rewards = train_q_learning(env, q_agent, num_episodes=1000)
    sarsa_rewards = train_sarsa(env, sarsa_agent, num_episodes=1000)
    
    # Save the trained Q-tables
    np.save("q_table_qlearning.npy", q_agent.q_table)
    np.save("q_table_sarsa.npy", sarsa_agent.q_table)
    print("Saved Q-tables to 'q_table_qlearning.npy' and 'q_table_sarsa.npy'.")
    
    # Evaluate strategies
    num_eval_episodes = 100
    results = {}
    
    print("\nEvaluating policies...")
    results["Trained Q-Learning"] = evaluate_policy(env, q_agent, "greedy", num_eval_episodes)
    results["Trained SARSA"] = evaluate_policy(env, sarsa_agent, "greedy", num_eval_episodes)
    results["Longest Queue First (LQF)"] = evaluate_policy(env, None, "lqf", num_eval_episodes)
    results["Fixed-Time (5 steps)"] = evaluate_policy(env, None, "fixed", num_eval_episodes, switch_interval=5)
    results["Fixed-Time (10 steps)"] = evaluate_policy(env, None, "fixed", num_eval_episodes, switch_interval=10)
    results["Random Switch"] = evaluate_policy(env, None, "random", num_eval_episodes)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<30}{'Avg Episode Reward':<25}{'Avg Waiting Cars/Step'}")
    print("=" * 80)
    for strategy, (avg_reward, avg_cars) in results.items():
        print(f"{strategy:<30}{avg_reward:<25.2f}{avg_cars:<25.2f}")
    print("=" * 80 + "\n")
    
    # Plot 1: Learning curves
    plt.figure(figsize=(11, 6))
    window = 50
    
    # Raw rewards with transparency
    plt.plot(q_rewards, alpha=0.15, color='#1e88e5', label='Q-Learning Raw')
    plt.plot(sarsa_rewards, alpha=0.15, color='#ffb300', label='SARSA Raw')
    
    # Moving averages
    q_ma = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    sarsa_ma = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(np.arange(window-1, len(q_rewards)), q_ma, color='#0d47a1', linewidth=2.5, label='Q-Learning (Moving Avg)')
    plt.plot(np.arange(window-1, len(sarsa_rewards)), sarsa_ma, color='#e65100', linewidth=2.5, label='SARSA (Moving Avg)')
    
    plt.title("Learning Curves (Q-Learning vs. SARSA)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Episode Reward", fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=300)
    plt.close()
    print("Saved 'learning_curves.png'.")
    
    # Plot 2: Bar chart comparison
    strategies = list(results.keys())
    rewards = [results[s][0] for s in strategies]
    cars = [results[s][1] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1565c0', '#d84315', '#2e7d32', '#f57f17', '#6a1b9a', '#c62828']
    
    # Plot rewards
    bars1 = ax1.bar(strategies, rewards, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_title("Average Episode Reward (Higher is Better)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel("Reward", fontsize=11)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval - (5 if yval < 0 else -5), f"{yval:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    # Plot average waiting cars
    bars2 = ax2.bar(strategies, cars, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_title("Average Waiting Cars/Step (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Cars", fontsize=11)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.suptitle("Algorithm Evaluation & Baseline Comparison", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("policy_comparison.png", dpi=300)
    plt.close()
    print("Saved 'policy_comparison.png'.")

if __name__ == "__main__":
    run_comparison()
