import torch
from torch import nn
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()

        n_actions = action_dim[0]

        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(state_dim)
        print(conv_out_size)

        # Estimates the parameters of a Beta distribution over actions
        self.actor_fc = nn.Sequential(nn.Linear(conv_out_size, 256), nn.ReLU(),)

        self.alpha_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())

        # Estimates the value of the state
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 256), nn.ReLU(), nn.Linear(256, 1), nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)

        # Estimate value of the state
        value = self.critic(x)

        # Estimate the parameters of a Beta distribution over actions
        x = self.actor_fc(x)

        # add 1 to alpha & beta to ensure the distribution is "concave and unimodal" (https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return value, alpha, beta

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv(x)

        return int(np.prod(x.size()))

