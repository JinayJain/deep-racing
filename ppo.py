import gym
import torch
from torch import optim, nn
import torchvision.transforms as T
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

from net import Actor, Critic

"""
TODO: Look more through https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py

PPO Algorithm:

1. Initialize policy and value function networks (Pi and V)
2. For some number of iterations:
    a. Use the current policy to generate trajectories
    b. Compute the rewards-to-go for each trajectory
"""

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


class PPO:
    def __init__(
        self,
        env: gym.Env,
        lr=5e-3,
        num_batches=5000,
        batch_timesteps=1024,
        max_episode_timesteps=512,
        updates_per_batch=15,
        gamma=0.99,
        variance=0.4,
        clip=0.2,
    ):
        self.env = env
        self.lr = lr
        self.batch_timesteps = batch_timesteps
        self.updates_per_batch = updates_per_batch
        self.max_episode_timesteps = max_episode_timesteps
        self.gamma = gamma
        self.clip = clip
        self.num_batches = num_batches

        self.preprocess = self._make_preprocess()

        state_sample = self.preprocess(self.env.observation_space.sample())
        action_sample = env.action_space.sample()
        print(action_sample, state_sample.shape)
        self.actor = Actor(state_sample.shape[0], action_sample.shape[0]).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_sample.shape[0]).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        cov_vector = torch.full(action_sample.shape, variance)
        self.cov_matrix = torch.diag(cov_vector)

    def train(self):
        """
        Train the PPO model
        """

        for b in range(self.num_batches):
            # Compute advantage with "old" policy
            with torch.no_grad():
                (
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_rtgs,
                    batch_ep_rewards,
                ) = self.sample_trajectories()

                V = self.critic(batch_states).squeeze(1)
                adv = batch_rtgs - V
                adv = (adv - adv.mean()) / adv.std()

            print(sum(batch_ep_rewards) / len(batch_ep_rewards))

            for i in range(self.updates_per_batch):
                # Get the current policy's action and value estimates
                means = self.actor(batch_states)
                curr_policy = MultivariateNormal(
                    means, covariance_matrix=self.cov_matrix.to(device)
                )

                # Compute the clipped surrogate policy loss from PPO paper
                curr_log_probs = curr_policy.log_prob(batch_actions)
                policy_ratio = (curr_log_probs - batch_log_probs).exp()

                # Compute clipped surrogate loss function
                policy_loss_raw = adv * policy_ratio
                policy_loss_clipped = (
                    torch.clamp(policy_ratio, 1 - self.clip, 1 + self.clip) * adv
                )

                print((policy_loss_raw - policy_loss_clipped).abs().sum())

                V = self.critic(batch_states).squeeze(1)

                policy_loss = (-torch.min(policy_loss_raw, policy_loss_clipped)).mean()
                value_loss = nn.MSELoss()(V.squeeze(), batch_rtgs)

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

        first_episode = True
        while t < self.batch_timesteps:
            state = self.preprocess(self.env.reset()).to(device)

            episode_rewards = []

            total_reward = 0
            for i in range(self.max_episode_timesteps):
                t += 1

                action, log_prob = self.get_action(state)

                next_frame, reward, done, _ = self.env.step(action.numpy())
                total_reward += reward

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                episode_rewards.append(reward)

                state = self.preprocess(next_frame).to(device)

                if done:
                    break

                if first_episode:
                    self.env.render()

            for i in reversed(range(len(episode_rewards) - 1)):
                episode_rewards[i] += self.gamma * episode_rewards[i + 1]

            batch_rtgs += episode_rewards
            batch_ep_rewards.append(total_reward)

            first_episode = False

        # plt.plot(batch_rtgs)
        # plt.plot(batch_values)
        # plt.show()

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
            means = self.actor(state).cpu()

            policy = MultivariateNormal(means, covariance_matrix=self.cov_matrix)
            action = policy.sample()

        return action, policy.log_prob(action)

    def _make_preprocess(self):
        # return T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])
        def preprocess(x):
            return torch.tensor(x, dtype=torch.float32)

        return preprocess

