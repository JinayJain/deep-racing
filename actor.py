from torch import nn


class Actor(nn.Module):
    def __init__(self) -> None:
        super(Actor, self).__init__()
