import numpy as np
import random
from typing import Tuple

class QLearningAgent:
    """
    A Q-Learning agent that learns to optimize traffic signal switching.
    
    The agent maintains a Q-table of shape (ns_states, ew_states, light_states, action_states)
    to estimate the expected future rewards for taking actions in different traffic scenarios.
    """
    
    def __init__(
        self, 
        max_queue: int = 5, 
        num_lights: int = 2, 
        num_actions: int = 2
    ):
        """
        Initializes the Q-Learning Agent.
        
        Args:
            max_queue (int): Capped length of queues (states range from 0 to max_queue).
            num_lights (int): Number of light configurations (0 or 1).
            num_actions (int): Number of actions (0: Keep, 1: Switch).
        """
        self.num_actions = num_actions
        
        # Dimensions of state space (max_queue + 1 for indices 0 to max_queue)
        state_dimensions = (max_queue + 1, max_queue + 1, num_lights)
        
        # Q-table shape: (NS_queue_len, EW_queue_len, Light_phase, Action)
        q_table_shape = state_dimensions + (num_actions,)
        
        # Initialize the Q-table with zeros
        self.q_table = np.zeros(q_table_shape)

    def choose_action(self, state: Tuple[int, int, int], epsilon: float) -> int:
        """
        Chooses an action using the Epsilon-Greedy strategy.
        
        Args:
            state (Tuple[int, int, int]): The current state (ns_queue, ew_queue, green_light).
            epsilon (float): Exploration rate, probability of choosing a random action.
            
        Returns:
            action (int): Selected action (0: keep current light, 1: switch light).
        """
        ns_queue, ew_queue, green_light = state
        
        # Exploration: Choose a random action
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Exploitation: Choose the best action from the Q-table
        q_values = self.q_table[ns_queue, ew_queue, green_light]
        
        # Random tie-breaking for better learning dynamics (especially with all zeros at start)
        max_q = np.max(q_values)
        actions_with_max_q = np.flatnonzero(q_values == max_q)
        return int(random.choice(actions_with_max_q))

    def update_q_table(
        self, 
        state: Tuple[int, int, int], 
        action: int, 
        reward: float, 
        next_state: Tuple[int, int, int], 
        alpha: float, 
        gamma: float
    ) -> None:
        """
        Updates the Q-table using the standard Q-learning temporal difference update (Bellman equation).
        
        Formula:
            Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
            
        Args:
            state (Tuple[int, int, int]): The current state before action.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (Tuple[int, int, int]): The state transition occurred.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        """
        ns, ew, light = state
        next_ns, next_ew, next_light = next_state
        
        # Get current estimate
        current_q = self.q_table[ns, ew, light, action]
        
        # Get maximum estimate for the next state
        max_future_q = np.max(self.q_table[next_ns, next_ew, next_light])
        
        # Calculate the target
        td_target = reward + gamma * max_future_q
        
        # Temporal difference error
        td_error = td_target - current_q
        
        # Update the Q-table
        self.q_table[ns, ew, light, action] = current_q + alpha * td_error

class SARSAAgent(QLearningAgent):
    """
    A SARSA (State-Action-Reward-State-Action) agent that learns to optimize traffic signal switching.
    
    SARSA is an on-policy RL algorithm, meaning it updates its Q-table
    using the action actually selected in the next state under the current policy,
    rather than taking the maximum Q-value.
    """
    
    def update_q_table(
        self, 
        state: Tuple[int, int, int], 
        action: int, 
        reward: float, 
        next_state: Tuple[int, int, int], 
        next_action: int,
        alpha: float, 
        gamma: float
    ) -> None:
        """
        Updates the Q-table using the SARSA update rule:
        Q(s, a) = Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
        
        Args:
            state (Tuple[int, int, int]): The current state before action.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (Tuple[int, int, int]): The state transition occurred.
            next_action (int): The action chosen in the next state.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
        """
        ns, ew, light = state
        next_ns, next_ew, next_light = next_state
        
        # Get current estimate
        current_q = self.q_table[ns, ew, light, action]
        
        # Get estimated value of the next state-action pair (on-policy)
        next_q = self.q_table[next_ns, next_ew, next_light, next_action]
        
        # Calculate the temporal difference target
        td_target = reward + gamma * next_q
        
        # Temporal difference error
        td_error = td_target - current_q
        
        # Update the Q-table
        self.q_table[ns, ew, light, action] = current_q + alpha * td_error

