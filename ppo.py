import gym
import torch
from torch import nn, optim
from torch.distributions import Beta
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(
        self,
        env: gym.Env,
        net: nn.Module,
        lr: float = 1e-4,
        batch_size: int = 32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        horizon: int = 256,
        epochs_per_step: int = 10,
        num_steps: int = 1000,
    ) -> None:
        self.env = env
        self.net = net.to(device)

        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.horizon = horizon
        self.epochs_per_step = epochs_per_step
        self.num_steps = num_steps
        self.gae_lambda = gae_lambda

        self.state = self._to_tensor(env.reset())

    def train(self):
        for step in range(self.num_steps):
            # Collect episode trajectory for the horizon length
            with torch.no_grad():
                self.collect_trajectory(self.horizon)

    def collect_trajectory(self, num_steps: int):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        for t in range(num_steps):
            # Run one step of the environment based on the current policy
            value, alpha, beta = self.net(self.state)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)

            policy = Beta(alpha, beta)
            action = policy.sample()
            log_prob = policy.log_prob(action)

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())

            if done:
                next_state = self.env.reset()

            next_state = self._to_tensor(next_state)

            # Store the transition
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            self.state = next_state

            self.env.render()

        # Get value of last state (used in GAE)
        final_value, _, _ = self.net(self.state)
        final_value = final_value.squeeze(0)

        # Compute generalized advantage estimates
        advantages = self._compute_gae(rewards, values, dones, final_value)

    def _compute_gae(self, rewards, values, dones, last_value):
        advantages = [0] * len(rewards)

        last_advantage = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (1 - dones[i]) * self.gamma * last_value - values[i]
            advantages[i] = (
                delta + (1 - dones[i]) * self.gamma * self.gae_lambda * last_advantage
            )

            last_value = values[i]
            last_advantage = advantages[i]

        return advantages

    def _to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

