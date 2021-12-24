import torch
from torch import nn
import gym
import numpy as np


class BipedalWalker(gym.ActionWrapper):
    def __init__(self) -> None:
        env = gym.make("LunarLanderContinuous-v2")
        super().__init__(env)

    def action(self, action):
        return action * 2 - 1


class BipedalNet(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()

        self.backbone = nn.Sequential(nn.Linear(state_dim[0], 256), nn.ReLU(),)

        self.actor_fc = nn.Linear(256, 256)
        self.alpha_head = nn.Sequential(nn.Linear(256, action_dim[0]), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(256, action_dim[0]), nn.Softplus())

        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.backbone(x)

        # Estimate value of the state
        value = self.critic(x)

        # Estimate the parameters of a Beta distribution over actions
        x = self.actor_fc(x)

        # add 1 to alpha & beta to ensure the distribution is "concave and unimodal" (https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return value, alpha, beta

