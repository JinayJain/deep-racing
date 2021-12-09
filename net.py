import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()

        self.backbone = Backbone()
        self.fc1 = nn.Linear(1152, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply tanh to first, sigmoid to remaining
        # Action space is ([-1, 1], [0, 1], [0, 1], [0, 1])
        x[:, 0] = torch.tanh(x[:, 0])
        x[:, 1:] = torch.sigmoid(x[:, 1:])

        return x


class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic, self).__init__()

        self.backbone = Backbone()
        self.fc1 = nn.Linear(1152 + 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.backbone(state)
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding="same")
        self.pool1 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        return x

