import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    """
    A Deep Q-Network (DQN) agent that controls traffic signal switching.
    Uses experience replay and a target network for stability.
    """
    def __init__(
        self, 
        state_dim: int = 6, 
        action_dim: int = 2, 
        lr: float = 1e-3, 
        gamma: float = 0.9, 
        buffer_size: int = 10000, 
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Verify torch is installed
        if torch is None:
            raise ImportError("PyTorch is not installed. Please run pip install torch first.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Initializing DQN Agent on device: {self.device}")
        
        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # Optimizer & Loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        # Replay memory
        self.memory = deque(maxlen=buffer_size)

    def preprocess_state(self, state) -> np.ndarray:
        """
        Converts the state tuple (ns_queue, ew_queue, green_light) into a normalized 
        continuous vector of size 6.
        - ns_queue (0-5) -> normalized by 5
        - ew_queue (0-5) -> normalized by 5
        - green_light (0-3) -> one-hot encoded vector of size 4
        """
        ns, ew, light = state
        
        # Normalized queues
        ns_norm = ns / 5.0
        ew_norm = ew / 5.0
        
        # One-hot light phase (4 possible states: 0, 1, 2, 3)
        light_onehot = [0.0, 0.0, 0.0, 0.0]
        if 0 <= light < 4:
            light_onehot[light] = 1.0
            
        state_vec = np.array([ns_norm, ew_norm] + light_onehot, dtype=np.float32)
        return state_vec

    def choose_action(self, state, epsilon: float) -> int:
        """
        Selects action using epsilon-greedy exploration.
        """
        # Exploration: random action
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
            
        # Exploitation: forward pass through network
        state_vec = self.preprocess_state(state)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            
        action = torch.argmax(q_values, dim=1).item()
        return int(action)

    def remember(self, state, action, reward, next_state, done):
        """Saves a transition to experience replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> float:
        """
        Samples a batch from memory and performs a single optimization step.
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample mini-batch
        batch = random.sample(self.memory, self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Preprocess states
        state_vecs = np.array([self.preprocess_state(s) for s in states], dtype=np.float32)
        next_state_vecs = np.array([self.preprocess_state(s) for s in next_states], dtype=np.float32)
        
        # Convert to Tensors
        states_tensor = torch.tensor(state_vecs, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(next_state_vecs, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        self.q_net.train()
        
        # Current Q-values
        current_q = self.q_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Target Q-values using standard DQN
        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + (self.gamma * max_next_q * (1.0 - dones_tensor))
            
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        """Copies weights from Q-Network to Target Network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, filepath):
        """Saves the network state dict."""
        torch.save(self.q_net.state_dict(), filepath)
        print(f"Saved DQN weights to '{filepath}'")

    def load(self, filepath):
        """Loads the network state dict."""
        self.q_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()
        print(f"Loaded DQN weights from '{filepath}'")
