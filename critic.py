from torch import nn


class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic, self).__init__()
