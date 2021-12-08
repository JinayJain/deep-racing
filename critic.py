import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(1280 + 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
