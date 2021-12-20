import torch
from torch import nn
import numpy as np


# class Net(nn.Module):
#     def __init__(self, state_shape, action_shape):
#         """
#         Combined network for policy and value function
#         """
#         super().__init__()

#         self.backbone = nn.Sequential(
#             nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         conv_out_size = self._get_conv_out(state_shape)

#         # Predicts the value of the state
#         self.value_head = nn.Sequential(
#             nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1),
#         )

#         # Predicts the means for the Gaussian distribution of the actions
#         self.policy_head = nn.Sequential(
#             nn.Linear(conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, action_shape[0]),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         conv_out = self.backbone(x)
#         value = self.value_head(conv_out)
#         means = self.policy_head(conv_out)
#         return value, means

#     def _get_conv_out(self, shape):
#         o = self.backbone(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()

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

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        return self.fc(conv_out)


class Critic(nn.Module):
    def __init__(self, state_dim) -> None:
        super().__init__()

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

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        return self.fc(conv_out)
