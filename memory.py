from collections import deque, namedtuple
import random

# S, A, R, S'
Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def push(self, state, action, reward, next_state):
        self.buffer.append(Transition(state, action, reward, next_state))

    def __len__(self):
        return len(self.buffer)
