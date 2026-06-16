import random
from typing import Tuple, Dict, Any

class TrafficEnv:
    """
    A simple Reinforcement Learning environment representing a 4-way intersection.
    
    State:
        A tuple of (ns_queue, ew_queue, green_light)
        - ns_queue (int): Number of cars waiting in the North/South lane (0 to 5).
        - ew_queue (int): Number of cars waiting in the East/West lane (0 to 5).
        - green_light (int): Current active green light direction. 
                             0 = North/South has green light.
                             1 = East/West has green light.
                             
    Actions:
        0: Keep current green light.
        1: Switch the green light (NS -> EW or EW -> NS).
        
    Reward (configurable via reward_mode):
        - "linear"    : -(ns_queue + ew_queue)        — minimize total waiting cars (default)
        - "max_queue" : -max(ns_queue, ew_queue)       — minimize the worst bottleneck
        - "quadratic" : -(ns_queue² + ew_queue²) / 25 — heavier penalty for large queues
        - "throughput": departures_ns + departures_ew  — reward actual cars that pass
    """
    
    REWARD_MODES = ("linear", "max_queue", "quadratic", "throughput")

    def __init__(self, max_queue: int = 5, max_steps: int = 100, reward_mode: str = "linear"):
        if reward_mode not in self.REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {self.REWARD_MODES}, got '{reward_mode}'")
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        
        # State variables
        self.ns_queue = 0
        self.ew_queue = 0
        self.green_light = 0  # 0: NS green, 1: EW green
        self.steps = 0
        self.time_since_switch = 3
        
        self.reset()

    def reset(self) -> Tuple[int, int, int]:
        """
        Resets the environment to a random initial state.
        
        Returns:
            Tuple[int, int, int]: The initial state (ns_queue, ew_queue, green_light).
        """
        # Start with random queue sizes between 0 and max_queue
        self.ns_queue = random.randint(0, self.max_queue)
        self.ew_queue = random.randint(0, self.max_queue)
        
        # Randomly choose initial green light direction (0 or 1)
        self.green_light = random.choice([0, 1])
        
        self.steps = 0
        self.time_since_switch = 3
        return self._get_state()

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """
        Advances the environment by one step given an action.
        
        Args:
            action (int): 0 to keep the light, 1 to switch the light.
            
        Returns:
            state (Tuple[int, int, int]): The next state.
            reward (float): The negative sum of waiting cars.
            done (bool): Whether the episode has finished.
            info (dict): Diagnostic info.
        """
        # We have 4 light states:
        # 0: NS Green, 1: EW Green
        # 2: NS Yellow (transitioning from 0 -> 1)
        # 3: EW Yellow (transitioning from 1 -> 0)
        
        departures_ns = 0
        departures_ew = 0
        
        if self.green_light in [2, 3]:
            # Yellow light phase automatically transitions to green in the next phase
            prev_light = self.green_light
            if prev_light == 2:
                self.green_light = 1  # NS Yellow -> EW Green
                # EW departures occur now that it is green for EW
                if self.ew_queue > 0:
                    departures_ew = min(self.ew_queue, random.choice([1, 2]))
            else:
                self.green_light = 0  # EW Yellow -> NS Green
                # NS departures occur
                if self.ns_queue > 0:
                    departures_ns = min(self.ns_queue, random.choice([1, 2]))
            self.time_since_switch = 0
            
        else:  # Active Green light phase
            if action == 1 and self.time_since_switch >= 3:
                # Transition to corresponding yellow phase
                if self.green_light == 0:
                    self.green_light = 2  # NS Green -> NS Yellow
                else:
                    self.green_light = 3  # EW Green -> EW Yellow
                # No departures can occur during a yellow phase
                self.time_since_switch = 0
            else:
                # Keep active green phase
                self.time_since_switch += 1
                if self.green_light == 0:  # NS Green
                    if self.ns_queue > 0:
                        departures_ns = min(self.ns_queue, random.choice([1, 2]))
                else:  # EW Green
                    if self.ew_queue > 0:
                        departures_ew = min(self.ew_queue, random.choice([1, 2]))
                        
        # 2. Simulate random arrivals (unequal traffic flow, dynamic demand phases)
        p_ns, p_ew = self.get_arrival_probabilities()
        arrivals_ns = 1 if random.random() < p_ns else 0
        arrivals_ew = 1 if random.random() < p_ew else 0
        
        # 3. Update the queue lengths (clamped between 0 and max_queue)
        self.ns_queue = min(self.max_queue, max(0, self.ns_queue - departures_ns + arrivals_ns))
        self.ew_queue = min(self.max_queue, max(0, self.ew_queue - departures_ew + arrivals_ew))
        
        # 4. Calculate reward based on the configured reward mode
        if self.reward_mode == "linear":
            reward = float(-(self.ns_queue + self.ew_queue))
        elif self.reward_mode == "max_queue":
            reward = float(-max(self.ns_queue, self.ew_queue))
        elif self.reward_mode == "quadratic":
            # No normalization: gives a strong signal that heavily penalizes large queues.
            # Range: 0 to -(max_queue² + max_queue²) = 0 to -50
            reward = float(-(self.ns_queue ** 2 + self.ew_queue ** 2))
        elif self.reward_mode == "throughput":
            # Hybrid: reward cars that actually pass, but still lightly penalize
            # queue buildup to avoid starvation. Keeps the reward in a negative-dominant
            # range consistent with the other modes.
            reward = float((departures_ns + departures_ew) - 0.1 * (self.ns_queue + self.ew_queue))
        else:
            reward = float(-(self.ns_queue + self.ew_queue))  # fallback
        
        # 5. Check termination condition
        self.steps += 1
        done = self.steps >= self.max_steps
        
        # 6. Get state and info
        state = self._get_state()
        info = {
            "departures": (departures_ns, departures_ew),
            "arrivals": (arrivals_ns, arrivals_ew),
            "phase": self.get_traffic_phase(),
            "rates": (p_ns, p_ew)
        }
        
        return state, reward, done, info

    def _get_state(self) -> Tuple[int, int, int]:
        """Helper method to return the state tuple."""
        return (self.ns_queue, self.ew_queue, self.green_light)

    def get_arrival_probabilities(self) -> Tuple[float, float]:
        """Returns the (ns_arrival_prob, ew_arrival_prob) based on current steps."""
        if self.steps < 30:
            return 0.8, 0.1  # Morning Rush (NS heavy)
        elif self.steps < 70:
            return 0.3, 0.3  # Mid-day (Balanced)
        else:
            return 0.1, 0.8  # Evening Rush (EW heavy)

    def get_traffic_phase(self) -> str:
        """Returns the name of the current traffic phase."""
        if self.steps < 30:
            return "Morning Rush"
        elif self.steps < 70:
            return "Mid-day Balanced"
        else:
            return "Evening Rush"


