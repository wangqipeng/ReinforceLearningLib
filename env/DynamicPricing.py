import gym
from gym import spaces
import numpy as np

class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        # Define the action space: 0 = low price, 1 = medium price, 2 = high price
        self.action_space = spaces.Discrete(3)
        # Define the observation space: here, a simple one-dimensional state (inventory level)
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        # state: inventory level, time period, competitor prices, and market demand indicators
        self.state = np.array([50], dtype=np.float32)  # starting with 50 units in inventory
        self.max_steps = 10
        self.current_step = 0
       
    def step(self, action):
        # Define price levels corresponding to actions
        price_levels = [10, 20, 30]
        price = price_levels[action]
        # Simulate demand (randomized for simplicity)
        # The transition is stochastic because the demand is randomly generated, meaning that 
        # the next state 𝑠′is probabilistic.
        demand = np.random.randint(0, 30)
        # Adjust demand based on pricing: lower price might boost demand
        if action == 0:
            demand = int(demand * 1.2)
        elif action == 2:
            demand = int(demand * 0.8)
        
        inventory = self.state[0]
        # Actual sales cannot exceed available inventory
        sales = min(demand, inventory)
        # Revenue is calculated as sales multiplied by the chosen price
        # Reward Function
        revenue = sales * price
        # Update inventory
        inventory -= sales
        self.state[0] = inventory
        
        # Reward is revenue; episode ends if we run out of inventory or reach max steps
        reward = revenue
        self.current_step += 1
        done = self.current_step >= self.max_steps or inventory <= 0
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([50], dtype=np.float32)
        self.current_step = 0
        return self.state

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Inventory: {self.state[0]}")