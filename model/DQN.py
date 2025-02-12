import torch
import math
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from config import EPSILON_START, EPSILON_END, EPSILON_DECAY, GAMMA, BATCH_SIZE, LEARNING_RATE

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#make the samples independent and identically distributed
class ReplayBuffer:
    def __init__(self, capacity= 10000):
        self.buffer = deque(maxlen = capacity)
    
    def push(self, state, action, rewards, next_state, done):
        self.buffer.append((state, action, rewards, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_state, done = zip(*samples)
        return np.array(states), actions, rewards, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size, device):# gamma = 0.99, lr = 0.001, epsilon = 1.0, epsilon_decay=0.995, min_epsilon = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.replay_buffer = ReplayBuffer()
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = LEARNING_RATE)
        self.count = 0
        self.criterion = nn.MSELoss()
        self.epsilon_threshold = 0.1

    def select_action(self, state):
        self.epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.count / EPSILON_DECAY) 
        self.count+=1
        if np.random.rand() < self.epsilon_threshold:
            action = np.random.choice(self.action_size)
        else:
            state_tensor = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.policy_net(state_tensor).argmax().item()
        return action

    def train(self, batch_size = 32):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
 
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states).gather(1, actions.type(torch.int64).unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()





