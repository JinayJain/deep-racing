from collections import namedtuple

import torch
from torch import nn, optim
import numpy as np
import random

from torch.nn.modules import conv

from util.replay import Memory

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class DQN:
    def __init__(
        self,
        input_shape,
        n_actions,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
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

        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.net.parameters(), lr=lr)

        self.input_shape = input_shape
        self.n_actions = n_actions

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

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

        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards).to(self.device).unsqueeze(1)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)

        q_pred = self.net(states).gather(0, actions)

        with torch.no_grad():
            next_actions = self.net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(0, next_actions)

            q_target = rewards + self.gamma * next_q

        self.optim.zero_grad()

        loss = self.criterion(q_pred, q_target)
        loss.backward()

        self.optim.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.net.state_dict())


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        print(conv_out_size)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

