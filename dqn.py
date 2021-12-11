from collections import namedtuple

import torch
from torch import nn, optim
import numpy as np
import random
import os

from torch.nn.modules import conv

from util.replay import Memory

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state"))


class DQN:
    def __init__(
        self,
        input_shape,
        n_actions,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_steps=1000000,
        memory_size=10000,
        memory_alpha=0.4,
        memory_beta=0.3,
        device=device,
    ):
        print(device)
        self.net = Net(input_shape, n_actions).to(device)
        self.target_net = Net(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.device = device

        self.criterion = nn.MSELoss(reduction="none")
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

        self.input_shape = input_shape
        self.n_actions = n_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_step_size = (epsilon - epsilon_min) / epsilon_steps

        self.memory = Memory(memory_size, alpha=memory_alpha, beta=memory_beta)

    def get_action(self, state, use_epsilon=True):
        """
        Get the policy's action and its Q-value for a given state.
        """

        state = state.to(self.device).unsqueeze(0)
        q_values = self.net(state).squeeze()

        if use_epsilon and random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            action = q_values.argmax().item()

        return action, q_values.cpu()

    def td_error(self, q_pred, reward, next_state):
        next_action = self.get_action(next_state)[0]

        next_state = next_state.to(self.device).unsqueeze(0)
        next_q = self.target_net(next_state).squeeze()[next_action]

        q_target = reward + self.gamma * next_q

        return (q_target - q_pred).item()

    def remember(self, state, action, reward, next_state, error):
        self.memory.add(error, Transition(state, action, reward, next_state))

    def train(self, batch_size) -> float:
        """
        Train the DQN on a single batch from memory. Returns the loss on the batch.
        """
        if len(self.memory) < batch_size:
            return 0.0

        batch, idxs, is_weight = self.memory.sample(batch_size)
        is_weight = torch.FloatTensor(is_weight).to(self.device).unsqueeze(1)

        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        rewards = torch.tensor(rewards).to(self.device).unsqueeze(1)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            device=self.device,
            dtype=torch.bool,
        )
        next_states = [s for s in next_states if s is not None]
        next_states = torch.stack(next_states).to(self.device)

        q_target = rewards
        with torch.no_grad():
            next_actions = self.net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)

            q_target[non_final_mask] += self.gamma * next_q

        self.optim.zero_grad()

        q_pred = self.net(states).gather(1, actions)

        loss = self.criterion(q_pred, q_target)
        loss = (loss * is_weight).mean()

        loss.backward()

        self.optim.step()

        td_error = q_pred - q_target
        for i, idx in enumerate(idxs):
            self.memory.update(idx, td_error[i].item())

        return loss.item()

    def step_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon -
                           self.epsilon_step_size)

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        # Count number of parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(self.n_params)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
