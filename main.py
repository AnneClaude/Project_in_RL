import numpy as np
import random
from environment import TrafficEnv
from agent import QLearningAgent

def train_agent(
    env: TrafficEnv, 
    agent: QLearningAgent, 
    num_episodes: int = 1000, 
    alpha: float = 0.1, 
    gamma: float = 0.9, 
    initial_epsilon: float = 1.0, 
    min_epsilon: float = 0.05, 
    decay_rate: float = 0.995
):
    """
    Trains the Q-Learning Agent in the Traffic environment.
    
    Args:
        env (TrafficEnv): The traffic environment.
        agent (QLearningAgent): The Q-learning agent.
        num_episodes (int): Number of training episodes.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        initial_epsilon (float): Starting exploration rate.
        min_epsilon (float): Minimum exploration rate.
        decay_rate (float): Epsilon decay multiplier per episode.
    """
    print(f"Starting training for {num_episodes} episodes...")
    epsilon = initial_epsilon
    episode_rewards = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 1. Choose action (epsilon-greedy)
            action = agent.choose_action(state, epsilon)
            
            # 2. Step the environment
            next_state, reward, done, _ = env.step(action)
            
            # 3. Update Q-table
            agent.update_q_table(state, action, reward, next_state, alpha, gamma)
            
            state = next_state
            total_reward += reward
            
        episode_rewards.append(total_reward)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode:4d}/{num_episodes} | Avg Reward (last 100): {avg_reward:6.1f} | Epsilon: {epsilon:.3f}")
            
    print("Training finished!\n")
    return episode_rewards

def evaluate_policy(env: TrafficEnv, agent: QLearningAgent = None, mode: str = "greedy", num_episodes: int = 100):
    """
    Evaluates a policy in the environment to compare performance.
    
    Args:
        env (TrafficEnv): The traffic environment.
        agent (QLearningAgent): The Q-learning agent (required for "greedy" mode).
        mode (str): Evaluation mode - "greedy" (using Q-table), "random" (random actions), 
                    or "fixed" (switches the light phase every 5 steps).
        num_episodes (int): Number of episodes to run evaluation.
        
    Returns:
        avg_reward (float): Average reward per episode.
        avg_waiting_cars (float): Average number of waiting cars per time step.
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
            if mode == "greedy" and agent is not None:
                action = agent.choose_action(state, epsilon=0.0)
            elif mode == "fixed":
                # Fixed time switching: switch light phase every 5 steps
                action = 1 if step_count % 5 == 0 and step_count > 0 else 0
            else:  # random actions
                action = random.choice([0, 1])
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Record queue length
            ns, ew, _ = state
            total_waiting_cars += (ns + ew)
            total_steps += 1
            
            state = next_state
            step_count += 1
            
        total_rewards.append(episode_reward)
        
    avg_reward = np.mean(total_rewards)
    avg_waiting_cars = total_waiting_cars / total_steps
    return avg_reward, avg_waiting_cars

def print_sample_q_values(agent: QLearningAgent):
    """
    Prints representative Q-values to show what the agent learned.
    Shows the scenario when North/South is currently Green (light = 0).
    """
    print("Sample Q-values for Light State 0 (North/South is Green):")
    print(f"{'NS Queue':<10}{'EW Queue':<10}{'Q(Keep Green)':<15}{'Q(Switch)':<15}{'Agent Decision'}")
    print("-" * 65)
    
    # Display queue states to demonstrate smart switching
    for ns in [0, 2, 4]:
        for ew in [0, 2, 4]:
            q_vals = agent.q_table[ns, ew, 0]
            best_action = np.argmax(q_vals)
            action_desc = "Keep NS Green" if best_action == 0 else "Switch to EW Green"
            print(f"{ns:<10}{ew:<10}{q_vals[0]:<15.2f}{q_vals[1]:<15.2f}{action_desc}")
    print()

if __name__ == "__main__":
    # Seed random number generators for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # 1. Initialize environment and agent
    env = TrafficEnv(max_queue=5, max_steps=100)
    agent = QLearningAgent(max_queue=5)
    
    # 2. Train agent
    train_agent(env, agent, num_episodes=1000)
    
    # 3. Evaluate and compare policies
    print("Evaluating policies (averaged over 100 episodes):")
    greedy_reward, greedy_cars = evaluate_policy(env, agent, mode="greedy", num_episodes=100)
    fixed_reward, fixed_cars = evaluate_policy(env, None, mode="fixed", num_episodes=100)
    random_reward, random_cars = evaluate_policy(env, None, mode="random", num_episodes=100)
    
    print(f"{'Strategy':<22}{'Avg Episode Reward':<22}{'Avg Waiting Cars/Step'}")
    print("-" * 65)
    print(f"{'Trained Q-Learning':<22}{greedy_reward:<22.2f}{greedy_cars:<20.2f}")
    print(f"{'Fixed-Time Switch':<22}{fixed_reward:<22.2f}{fixed_cars:<20.2f}")
    print(f"{'Random Switch':<22}{random_reward:<22.2f}{random_cars:<20.2f}")
    print("-" * 65)
    print()
    
    # 4. Print sample learned decisions
    print_sample_q_values(agent)
