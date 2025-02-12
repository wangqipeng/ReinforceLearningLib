import gym
import numpy as np
import torch
from DQN import DQNAgent

import matplotlib.pyplot as plt
# Define the Dynamic Pricing Environment
class DynamicPricingEnv(gym.Env):
    def __init__(self, max_time=100, max_inventory=100, min_price=5, max_price=50):
        super(DynamicPricingEnv, self).__init__()
        self.max_time = max_time
        self.time = 0
        self.inventory = max_inventory
        self.max_inventory = max_inventory
        self.min_price = min_price
        self.max_price = max_price
        self.price = np.random.uniform(min_price, max_price)

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, min_price]),
            high = np.array([max_time, max_inventory, max_price]),
            dtype=np.float32
        )

    def step(self, action):
        if action == 0:
            self.price = max(self.min_price, self.price - 1)
        elif action == 1:
            self.price = min(self.max_price, self.price + 1)

        #simulated demand function (high price -> lower demand)
        demand = max(0, self.max_inventory // (self.price / 5))
        sales = min(self.inventory, demand)
        self.inventory -= sales
        revenue = sales * self.price

        self.time += 1
        done = self.time >= self.max_time or self.inventory <= 0
        # Next State
        next_state = np.array([self.time, self.inventory, self.price], dtype=np.float32)
        return next_state, revenue, done, {}

    def reset(self):
        self.time = 0
        self.inventory = self.max_inventory
        self.price = np.random.uniform(self.min_price, self.max_price)
        return np.array([self.time, self.inventory, self.price], dtype=np.float32)

    def render(self):
        print(f"Time: {self.time}, Inventory: {self.inventory}, Price: {self.price}")

if __name__ == "__main__":
    env = DynamicPricingEnv()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = DQNAgent(state_size=3, action_size=3, device = device)
    env_name = "dynamic pricing"
    episodes = 30
    batch_size = 32
    return_list = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train(batch_size)

            state = next_state
            total_reward += reward
            if done:
                break
            return_list.append(total_reward)
        print(f"Episode {episode+1}, Total Revenue: {total_reward}, Epsilon: {agent.epsilon_threshold:.2f}")

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()
