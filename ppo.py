import gym
import torch
from torch import optim, nn
import torchvision.transforms as T
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from os import path

from net import Actor, Critic
from util.memory import Memory

"""
TODO: Look more through https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py

PPO Algorithm:

1. Initialize policy and value function networks (Pi and V)
2. For some number of iterations:
    a. Use the current policy to generate trajectories
    b. Compute the rewards-to-go for each trajectory
"""

# device = torch.device("cuda" if torch.cuda.is_available else "cpu")
device = torch.device("cpu")


class PPO:
    def __init__(
        self,
        env: gym.Env,
        lr=1e-3,
        num_batches=5000,
        epoch_timesteps=2048,
        max_episode_timesteps=1024,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        variance=0.2,
        clip=0.2,
        save_interval=50,
        render_interval=50,
    ):
        self.env = env
        self.lr = lr
        self.epoch_timesteps = epoch_timesteps
        self.n_epochs = n_epochs
        self.max_episode_timesteps = max_episode_timesteps
        self.gamma = gamma
        self.clip = clip
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.render_interval = render_interval

        self.preprocess = self._make_preprocess()

        state_sample = self.preprocess(self.env.observation_space.sample())
        action_sample = env.action_space.sample()
        print(action_sample, state_sample.shape)
        self.actor = Actor(state_sample.shape, action_sample.shape[0]).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_sample.shape).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        cov_vector = torch.full(action_sample.shape, variance)
        self.cov_matrix = torch.diag(cov_vector)

        self.n_episodes = 0

    def train(self):
        """
        Train the PPO model
        """

        for b in range(self.num_batches):
            # Compute advantage with "old" policy
            with torch.no_grad():
                (
                    full_states,
                    full_actions,
                    full_log_probs,
                    full_rtgs,
                    full_ep_rewards,
                ) = self.sample_trajectories()

                V = self.critic(full_states).squeeze(1)
                full_adv = full_rtgs - V
                full_adv = (full_adv - full_adv.mean()) / full_adv.std()

            print(sum(full_ep_rewards) / len(full_ep_rewards))

            # Assemble batched dataset
            memory = Memory(
                full_states, full_actions, full_log_probs, full_adv, full_rtgs
            )

            memory_loader = DataLoader(
                memory, batch_size=self.batch_size, shuffle=True, num_workers=0
            )

            for i in range(self.n_epochs):
                for (
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_adv,
                    batch_rtgs,
                ) in memory_loader:
                    # Get the current policy's action and value estimates
                    means = self.actor(batch_states)
                    curr_policy = MultivariateNormal(
                        means, covariance_matrix=self.cov_matrix.to(device)
                    )

                    # Compute the clipped surrogate policy loss from PPO paper
                    curr_log_probs = curr_policy.log_prob(batch_actions)
                    policy_ratio = (curr_log_probs - batch_log_probs).exp()

                    # Compute clipped surrogate loss function
                    policy_loss_raw = batch_adv * policy_ratio
                    policy_loss_clipped = (
                        torch.clamp(policy_ratio, 1 - self.clip, 1 + self.clip)
                        * batch_adv
                    )

                    # print((policy_loss_raw - policy_loss_clipped).abs().sum().item())

                    V = self.critic(batch_states).squeeze(1)

                    policy_loss = (
                        -torch.min(policy_loss_raw, policy_loss_clipped)
                    ).mean()
                    value_loss = nn.MSELoss()(V, batch_rtgs)

                    # Update networks
                    self.actor_optim.zero_grad()
                    self.critic_optim.zero_grad()

                    policy_loss.backward()
                    value_loss.backward()

                    self.actor_optim.step()
                    self.critic_optim.step()

    def sample_trajectories(self):
        """
        Sample trajectories from the environment, using the current policy
        """

        t = 0

        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_rtgs = []
        batch_ep_rewards = []

        batch_values = []

        first_episode = True
        while t < self.epoch_timesteps:
            state = self.preprocess(self.env.reset()).to(device)

            episode_rewards = []

            total_reward = 0
            self.n_episodes += 1

            for i in range(self.max_episode_timesteps):
                t += 1

                action, log_prob = self.get_action(state)
                value = self.critic(state.unsqueeze(0)).item()

                next_frame, reward, done, _ = self.env.step(action.numpy())
                total_reward += reward

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)

                episode_rewards.append(reward)

                state = self.preprocess(next_frame).to(device)

                if done:
                    break

                if self.n_episodes % self.render_interval == 0:
                    self.env.render()

            for i in reversed(range(len(episode_rewards) - 1)):
                episode_rewards[i] += self.gamma * episode_rewards[i + 1]

            batch_rtgs += episode_rewards
            batch_ep_rewards.append(total_reward)

            first_episode = False

            if self.n_episodes % self.save_interval == 0:
                self.save("ckpt", self.n_episodes)

        # Convert to tensors
        batch_states = torch.stack(batch_states).to(device).float().detach()
        batch_actions = torch.stack(batch_actions).to(device).float().detach()
        batch_log_probs = torch.tensor(batch_log_probs).to(device).float().detach()
        batch_rtgs = torch.tensor(batch_rtgs).to(device).float().detach()

        return (
            batch_states,
            batch_actions,
            batch_log_probs,
            batch_rtgs,
            batch_ep_rewards,
        )

    def get_action(self, state):
        with torch.no_grad():
            means = self.actor(state.unsqueeze(0)).cpu().squeeze(0)

            policy = MultivariateNormal(means, covariance_matrix=self.cov_matrix)
            action = policy.sample()

        return action, policy.log_prob(action)

    def _make_preprocess(self):
        return T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
        # def preprocess(x):
        #     return torch.tensor(x, dtype=torch.float32)

        # return preprocess

    def save(self, folder, episode):
        torch.save(self.actor.state_dict(), path.join(folder, f"{episode}_actor.pth"))
        torch.save(self.critic.state_dict(), path.join(folder, f"{episode}_critic.pth"))

    def load(self, folder, episode):
        self.actor.load_state_dict(
            torch.load(path.join(folder, f"{episode}_actor.pth"))
        )
        self.critic.load_state_dict(
            torch.load(path.join(folder, f"{episode}_critic.pth"))
        )

