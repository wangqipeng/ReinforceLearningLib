from DynamicPricing import DynamicPricingEnv
import random
import numpy as np

env = DynamicPricingEnv()

# Q-learning parameters
num_episodes = 5
alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor
epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.1    # Minimum exploration rate
epsilon_decay = 0.995

# Discretization for the continuous state (inventory from 0 to 100 into 11 bins)
num_bins = 11
q_table = np.zeros((num_bins, env.action_space.n))
print("===========begin============")
print(q_table)
def discretize(state):
    inventory = state[0]
    bin_index = int(inventory / 10)
    return min(bin_index, num_bins - 1)

rewards_per_episode = []

for episode in range(num_episodes):
    state = env.reset()
    state_idx = discretize(state)
    total_reward = 0
    done = False

    while not done:
        # Epsilon-greedy action selection for exploration/exploitation trade-off
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_idx])
        
        next_state, reward, done, _ = env.step(action)
        
        next_state_idx = discretize(next_state)

        # Q-learning update step
        best_next_action = np.argmax(q_table[next_state_idx])
        td_target = reward + gamma * q_table[next_state_idx, best_next_action]
        q_table[state_idx, action] += alpha * (td_target - q_table[state_idx, action])
        #print(q_table)
        state_idx = next_state_idx
        total_reward += reward
    
    # Decay exploration rate over episodes
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)
    
    # Optional: print progress every 50 episodes
    if episode % 50 == 0:
        print(f"Episode {episode}: Total Reward: {total_reward}")

print("Training finished.")


