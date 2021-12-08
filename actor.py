import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply tanh to first, sigmoid to remaining
        # Action space is ([-1, 1], [0, 1], [0, 1])
        x[:, 0] = torch.tanh(x[:, 0])
        x[:, 1:] = torch.sigmoid(x[:, 1:])

        return x
