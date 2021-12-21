import torch
from torch import nn
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(state_dim)

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        return self.actor(x), self.critic(x)

