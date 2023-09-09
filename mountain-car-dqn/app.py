import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque

env = gym.make("MountainCar-v0")
writer = SummaryWriter("runs/dqn_mountain_car")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.global_step = 0
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = QNetwork(state_size, action_size).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        return torch.argmax(action_values, dim=1).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Get max Q values for next states from model
        target_qvals = (
            rewards
            + (1 - dones) * self.gamma * self.model(next_states).max(1)[0].detach()
        )

        # Get current Q values for batch from model using predicted actions
        current_qvals = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute loss between current and target Q values
        loss = self.criterion(current_qvals, target_qvals.unsqueeze(1))

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        writer.add_scalar("Loss", loss.item(), self.global_step)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


EPISODES = 1000
agent = DQNAgent(state_size=2, action_size=3)
batch_size = 32

for e in range(EPISODES):
    state, info = env.reset(seed=42)
    total_reward = 0
    for time in range(200):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        reward = reward if not done else -10
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        writer.add_scalar("Reward", total_reward, e)
        writer.add_scalar("Epsilon", agent.epsilon, e)

        if done:
            print(
                f"episode: {e}/{EPISODES}, score: {time}, epsilon: {agent.epsilon:.2f}"
            )
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
env.close()

env = gym.make("MountainCar-v0", render_mode="human")
state, info = env.reset(seed=42)
while True:
    action = agent.act(state)
    next_state, reward, done, _, _ = env.step(action)
    state = next_state
    if done:
        break

writer.close()
