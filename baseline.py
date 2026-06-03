import numpy as np
import random
from environment import TrafficEnv

def run_baseline(num_episodes: int = 50, switch_interval: int = 10):
    """
    Runs a fixed-timer baseline for the Traffic environment.
    The green light is switched every `switch_interval` steps.
    
    Args:
        num_episodes (int): Number of episodes to run.
        switch_interval (int): How many steps to wait before switching the light.
        
    Returns:
        tuple: Average total reward and average waiting cars per step.
    """
    # Create the environment with the same standard parameters
    env = TrafficEnv(max_queue=5, max_steps=100)
    
    total_rewards = []
    total_waiting_cars = 0
    total_steps = 0
    
    print(f"Starting Baseline Evaluation over {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # Implement fixed timer logic: switch every 10 steps
            # This implicitly respects the 3-step minimum delay constraint since 10 > 3.
            if step_count > 0 and step_count % switch_interval == 0:
                action = 1  # Switch the light
            else:
                action = 0  # Keep the current light
                
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Record queue sizes
            ns, ew, _ = state
            total_waiting_cars += (ns + ew)
            total_steps += 1
            
            state = next_state
            step_count += 1
            
        total_rewards.append(episode_reward)
        
    avg_reward = np.mean(total_rewards)
    avg_waiting_cars = total_waiting_cars / total_steps
    
    print("\n========================================")
    print(f"BASELINE EVALUATION RESULTS")
    print("========================================")
    print(f"Strategy:              Switch light every {switch_interval} steps")
    print(f"Episodes:              {num_episodes}")
    print(f"Average Total Reward:  {avg_reward:.2f}")
    print(f"Avg Waiting Cars/Step: {avg_waiting_cars:.2f}")
    print("========================================\n")
    
    return avg_reward, avg_waiting_cars

if __name__ == "__main__":
    # Seed the random number generators so you get a consistent, direct 
    # comparison with the Q-learning agent's evaluation.
    random.seed(42)
    np.random.seed(42)
    
    run_baseline(num_episodes=50, switch_interval=10)
