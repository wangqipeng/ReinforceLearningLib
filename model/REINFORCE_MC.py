# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces
import random

# ------------------------------
# 1. Define the Dynamic Pricing Environment
# ------------------------------

class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        # Define the action space: 3 discrete actions (e.g., low, medium, high prices)
        self.action_space = spaces.Discrete(3)
        # Define the observation space: a single continuous value (inventory level from 0 to 100)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        # Initialize the state with an inventory of 50 units
        self.state = np.array([50], dtype=np.float32)
        # Maximum number of steps in one episode (finite horizon)
        self.max_steps = 10
        # Initialize the step counter
        self.current_step = 0

    def step(self, action):
        # Define three price levels corresponding to the three possible actions
        price_levels = [10, 20, 30]
        # Get the chosen price based on the action taken
        price = price_levels[action]
        
        # Simulate random demand (for simplicity, a random integer between 0 and 29)
        demand = np.random.randint(0, 30)
        # Adjust demand: lower price (action 0) boosts demand; higher price (action 2) reduces demand
        if action == 0:
            demand = int(demand * 1.2)
        elif action == 2:
            demand = int(demand * 0.8)
        
        # Current inventory is our state
        inventory = self.state[0]
        # Actual sales are limited by the available inventory
        sales = min(demand, inventory)
        # Revenue is the number of sales multiplied by the chosen price
        revenue = sales * price
        # Update inventory after sales
        inventory -= sales
        self.state[0] = inventory
        
        # Increment the step counter
        self.current_step += 1
        # Episode is done if maximum steps reached or inventory is depleted
        done = self.current_step >= self.max_steps or inventory <= 0
        
        # Return the new state, the reward (revenue), the done flag, and an empty info dict
        return self.state.copy(), revenue, done, {}

    def reset(self):
        # Reset state to initial conditions at the start of an episode
        self.state = np.array([50], dtype=np.float32)
        self.current_step = 0
        return self.state.copy()

    def render(self, mode='human'):
        # For debugging: print the current step and inventory
        print(f"Step: {self.current_step}, Inventory: {self.state[0]}")

# ------------------------------
# 2. Define the Policy Network
# ------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # Fully connected layer from input to hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Activation function (ReLU)
        self.relu = nn.ReLU()
        # Fully connected layer from hidden layer to output (action logits)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Softmax layer to convert logits to probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Pass input through first layer
        x = self.fc1(x)
        # Apply non-linearity
        x = self.relu(x)
        # Pass through second layer
        x = self.fc2(x)
        # Convert logits to probabilities over actions
        probs = self.softmax(x)
        return probs

# ------------------------------
# 3. Helper Function: Compute Returns
# ------------------------------

def compute_returns(rewards, gamma=0.99):
    """
    Computes discounted cumulative rewards for an episode.
    Args:
        rewards (list): Rewards collected during the episode.
        gamma (float): Discount factor.
    Returns:
        Tensor of normalized returns.
    """
    returns = []
    R = 0
    # Process rewards in reverse order to calculate cumulative sum
    for r in reversed(rewards):
        R = r + gamma * R  # Discount future rewards
        returns.insert(0, R)  # Insert at beginning to maintain order
    # Convert list to a PyTorch tensor
    returns = torch.tensor(returns)
    # Normalize returns for stability (optional but often helpful)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# ------------------------------
# 4. Training Loop Using the REINFORCE Algorithm
# ------------------------------

def train_policy_gradient(env, policy_net, optimizer, num_episodes=500, gamma=0.99):
    # Set the network to training mode
    policy_net.train()
    
    # Loop over episodes
    for episode in range(num_episodes):
        # Reset the environment for a new episode and get the initial state
        state = env.reset()
        # Lists to store log probabilities and rewards for the episode
        log_probs = []
        rewards = []
        done = False
        
        # Run one episode until termination
        while not done:
            # Convert the state (NumPy array) into a PyTorch tensor and add batch dimension
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            # Get the action probabilities from the policy network
            probs = policy_net(state_tensor)
            # Create a categorical distribution over the actions
            dist = torch.distributions.Categorical(probs)
            # Sample an action from the distribution
            action = dist.sample()
            # Get the log probability of the chosen action (used in the loss)
            log_prob = dist.log_prob(action)
            # Store the log probability for this time step
            log_probs.append(log_prob)
            
            # Take the action in the environment
            state, reward, done, _ = env.step(action.item())
            # Record the reward
            rewards.append(reward)
        
        # After the episode ends, compute the discounted returns for each time step
        returns = compute_returns(rewards, gamma)
        
        # Compute the policy loss:
        # For each time step, multiply the negative log probability by the return.
        # We want to adjust parameters so that actions leading to high returns become more probable.
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        # Sum over all time steps (the loss for the entire episode)
        policy_loss = torch.cat(policy_loss).sum()
        
        # Zero out gradients before backpropagation
        optimizer.zero_grad()
        # Backpropagate to compute gradients of the loss with respect to network parameters
        policy_loss.backward()
        # Update the network parameters using the optimizer
        optimizer.step()
        
        # Optionally, print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")

# ------------------------------
# 5. Main Execution: Initialize and Train the Agent
# ------------------------------

if __name__ == "__main__":
    # Create an instance of the dynamic pricing environment
    env = DynamicPricingEnv()
    
    # Set dimensions: input dimension is 1 (inventory level),
    # hidden dimension is an arbitrary choice (e.g., 128 neurons),
    # output dimension is the number of actions (3 price levels)
    input_dim = env.observation_space.shape[0]
    hidden_dim = 128
    output_dim = env.action_space.n
    
    # Instantiate the policy network
    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    
    # Set up an optimizer (Adam) for updating the policy network parameters
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    
    # Train the policy network using the REINFORCE algorithm
    train_policy_gradient(env, policy_net, optimizer, num_episodes=500, gamma=0.99)
