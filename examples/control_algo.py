# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:03:14 2023

@author: ritap
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample_batch(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

# Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output_activation(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_fc = nn.Linear(state_dim, 256)
        self.action_fc = nn.Linear(action_dim, 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, state, action):
        state_branch = self.activation(self.state_fc(state))
        action_branch = self.activation(self.action_fc(action))
        merged = torch.cat((state_branch, action_branch), dim=1)
        merged = self.activation(self.fc1(merged))
        output = self.fc2(merged)
        return output

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_high):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)

        self.buffer = ReplayBuffer(buffer_size=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        self.noise_std_dev = 0.5

    def get_action(self, state, exploration=True):
        state = torch.tensor(np.reshape(state, [1, self.state_dim]), dtype=torch.float32)
        action = self.actor(state)[0].detach().numpy()

        if exploration:
            noise = np.random.normal(0, self.noise_std_dev, size=self.action_dim)
            action += noise
        action = np.clip(action, -self.action_high, self.action_high)

        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.batch_size)

        target_actions = self.target_actor(next_states)
        target_q_values = self.target_critic(next_states, target_actions)
        target_values = rewards + self.gamma * (1 - dones) * target_q_values.detach()

        self.critic_optimizer.zero_grad()
        critic_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(critic_values, target_values)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        predicted_actions = self.actor(states)
        actor_loss = -torch.mean(self.critic(states, predicted_actions))
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

