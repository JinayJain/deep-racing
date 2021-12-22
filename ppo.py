import gym
import torch
from torch import nn, optim
from torch.distributions import Beta
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from logger import Logger

from memory import Memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(
        self,
        env: gym.Env,
        net: nn.Module,
        lr: float = 1e-4,
        batch_size: int = 128,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        horizon: int = 1024,
        epochs_per_step: int = 5,
        num_steps: int = 1000,
        clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
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
        self.clip = clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
        self.logger = Logger()

        self.state = self._to_tensor(env.reset())

    def train(self):
        for step in range(self.num_steps):
            # Collect episode trajectory for the horizon length
            with torch.no_grad():
                memory = self.collect_trajectory(self.horizon)

            memory_loader = DataLoader(
                memory, batch_size=self.batch_size, shuffle=True,
            )

            for epoch in range(self.epochs_per_step):
                for (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    advantages,
                    values,
                ) in memory_loader:
                    self.train_batch(
                        states, actions, log_probs, rewards, advantages, values
                    )
                    self.logger.print()

    def train_batch(
        self,
        states: torch.Tensor,
        old_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
    ):
        values, alpha, beta = self.net(states)
        values = values.squeeze(1)

        policy = Beta(alpha, beta)
        entropy = policy.entropy()
        log_probs = policy.log_prob(old_actions).sum(dim=1)

        ratio = (log_probs - old_log_probs).exp()  # same as policy / policy_old
        policy_loss_raw = ratio * advantages
        policy_loss_clip = (
            ratio.clamp(min=1 - self.clip, max=1 + self.clip) * advantages
        )
        policy_loss = -torch.min(policy_loss_raw, policy_loss_clip).mean()

        value_target = advantages + old_values  # V_t = (Q_t - V_t) + V_t
        value_loss = nn.MSELoss()(values, value_target)

        entropy_loss = -entropy.mean()

        loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        self.logger.log("Policy Loss", policy_loss.item())
        self.logger.log("Value Loss", value_loss.item())
        self.logger.log("Entropy Loss", entropy_loss.item())

        self.optim.zero_grad()

        loss.backward()

        self.optim.step()

        return loss.item(), policy_loss.item(), value_loss.item()

    def collect_trajectory(self, num_steps: int):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        for t in range(num_steps):
            # Run one step of the environment based on the current policy
            value, alpha, beta = self.net(self.state)
            value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)

            policy = Beta(alpha, beta)
            action = policy.sample()
            log_prob = policy.log_prob(action).sum()

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

        # Convert to tensors
        states = torch.cat(states)
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        values = torch.cat(values)

        # print shapes
        print("states:", states.shape)
        print("actions:", actions.shape)
        print("log_probs:", log_probs.shape)
        print("advantages:", advantages.shape)
        print("rewards:", rewards.shape)
        print("values:", values.shape)

        return Memory(states, actions, log_probs, rewards, advantages, values)

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

