from typing import Callable, Tuple
import gym
import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as T
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from os import path
import cv2
from collections import deque
from rich.progress import BarColumn, Progress, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from net import ActorCritic
from util.logger import CSVLogger, console
from util.memory import Memory

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


class PPO:
    def __init__(
        self,
        env: gym.Env,
        lr=5e-4,
        num_steps=5000,
        epoch_timesteps=1024,
        max_episode_timesteps=1000,
        reward_timeout=500,
        batch_size=128,
        n_epochs=5,
        gamma=0.99,
        variance=0.4,
        clip=0.2,
        save_interval=50,
        render_interval=1,
        value_coef=0.05,
        frame_stack=3,
        horizon=128,
    ):
        self.env = env
        self.lr = lr
        self.epoch_timesteps = epoch_timesteps
        self.n_epochs = n_epochs
        self.max_episode_timesteps = max_episode_timesteps
        self.gamma = gamma
        self.clip = clip
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.render_interval = render_interval
        self.reward_timeout = reward_timeout
        self.init_variance = variance
        self.value_coef = value_coef
        self.frame_stack_len = frame_stack

        self.n_episodes = 0
        self.flipped = False
        self.alpha = 1

        self.preprocess = self._make_preprocess()  # preprocesses states
        (
            self.n_actions,
            self.postprocess,
        ) = self._make_postprocess()  # postprocesses actions

        self.init_network()

        self.csv_logger = CSVLogger("summary.csv")
        self.progress = Progress()

        self.set_step_params()

    def init_network(self):
        state_sample = self.preprocess(self.env.observation_space.sample()).repeat(
            self.frame_stack_len, 1, 1
        )
        self.net = ActorCritic(state_sample.shape, self.n_actions).to(device)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optim, step_size=10, gamma=0.99)

    def set_step_params(self):
        self.alpha = self.scheduler.get_last_lr()[0] / self.lr

        self.variance = self.alpha * self.init_variance
        cov_vector = torch.full((self.n_actions,), self.variance)
        self.cov_matrix = torch.diag(cov_vector)

    def log_step_info(self, step, num_timesteps, avg_reward):
        console.rule(f"[b blue]Training Step {step}[/b blue]")
        console.log(f"Timesteps: {num_timesteps}")
        console.log(f"Average Reward: {avg_reward:.3f}")
        console.log(f"Alpha: {self.alpha:.3f}")
        console.log(f"Variance: {self.variance:.3f}")
        console.log(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.3e}")

    def train(self):
        """
        Train the PPO model
        """

        with Progress(
            "{task.description}",
            BarColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            "[{task.completed}/{task.total}]",
            console=console,
        ) as self.progress:
            step_task = self.progress.add_task("Step", total=self.num_steps)

            for step in range(self.num_steps):
                self.progress.advance(step_task)

                # Compute advantage with "old" policy
                with torch.no_grad():
                    # decay the variance and compute covariance matrix
                    self.set_step_params()

                    (
                        full_states,
                        full_actions,
                        full_log_probs,
                        full_rtgs,
                        full_ep_rewards,
                        full_adv,
                    ) = self.sample_trajectories()

                    _, values = self.net(full_states)
                    values = values.squeeze(1)

                    full_adv = full_rtgs - values
                    full_adv = (full_adv - full_adv.mean()) / full_adv.std()

                # Assemble batched dataset
                memory = Memory(
                    full_states, full_actions, full_log_probs, full_adv, full_rtgs
                )
                memory_loader = DataLoader(
                    memory, batch_size=self.batch_size, shuffle=True, num_workers=0
                )

                self.log_step_info(
                    step, len(memory), sum(full_ep_rewards) / len(full_ep_rewards)
                )

                epoch_task = self.progress.add_task("Epoch", total=self.n_epochs)
                for i in range(self.n_epochs):
                    self.progress.advance(epoch_task)

                    batch_task = self.progress.add_task(
                        f"Training on Batches", total=len(memory_loader)
                    )
                    for (
                        batch_states,
                        batch_actions,
                        batch_log_probs,
                        batch_adv,
                        batch_rtgs,
                    ) in memory_loader:
                        self.progress.advance(batch_task)

                        # Get the current policy's action and value estimates
                        actions, values = self.net(batch_states)
                        curr_policy = MultivariateNormal(
                            actions, covariance_matrix=self.cov_matrix.to(device)
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

                        policy_loss = (
                            -torch.min(policy_loss_raw, policy_loss_clipped)
                        ).mean()
                        values = values.squeeze(1)
                        value_loss = nn.SmoothL1Loss()(values, batch_rtgs)

                        combined_loss = policy_loss + self.value_coef * value_loss

                        # Update network
                        self.optim.zero_grad()

                        combined_loss.backward()

                        self.optim.step()

                    self.progress.remove_task(batch_task)

                self.progress.remove_task(epoch_task)
                self.scheduler.step()

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

        while t < self.epoch_timesteps:
            # randomly flip the environment (slowly decays over time)
            self.flipped = np.random.rand() < 0.5 * self.alpha

            episode_rewards = []

            total_reward = 0
            last_reward_step = 0
            self.n_episodes += 1

            frame = self.preprocess(self.env.reset()).to(device)

            frame_stack = deque(maxlen=self.frame_stack_len)
            frame_stack.append(frame)

            episode_task = self.progress.add_task(
                "Episode in Progress...", total=self.max_episode_timesteps,
            )

            for i in range(self.max_episode_timesteps):

                self.progress.advance(episode_task)
                t += 1

                # check if frame stack has enough frames
                if len(frame_stack) < self.frame_stack_len:
                    action = self.env.action_space.sample()
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess(next_state).to(device)
                    frame_stack.append(next_state)
                    continue

                state = torch.cat(tuple(frame_stack), dim=0).to(device)

                action, log_prob, value = self.get_action(state)

                next_frame, reward, done, _ = self.env.step(self.postprocess(action))
                total_reward += reward

                # clip reward
                reward = min(reward, 1)

                if reward > 0:
                    last_reward_step = i

                batch_states.append(state)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)

                episode_rewards.append(reward)

                frame = self.preprocess(next_frame).to(device)
                frame_stack.append(frame)

                if done or i - last_reward_step > self.reward_timeout:
                    break

                if self.n_episodes % self.render_interval == 0:
                    self.env.render()

            self.progress.remove_task(episode_task)

            for i in reversed(range(len(episode_rewards) - 1)):
                episode_rewards[i] += self.gamma * episode_rewards[i + 1]

            batch_rtgs += episode_rewards
            batch_ep_rewards.append(total_reward)

            self.csv_logger.write((self.n_episodes, total_reward))

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
            actions, value = self.net(state.unsqueeze(0))
            actions = actions.cpu().squeeze(0)

            policy = MultivariateNormal(actions, covariance_matrix=self.cov_matrix)
            action = policy.sample()

        return action, policy.log_prob(action), value

    def _make_preprocess(self) -> Callable:
        transform = T.Compose([T.ToPILImage(), T.Grayscale(), T.ToTensor()])

        def preprocess(x: np.ndarray) -> torch.Tensor:
            # remove score from screen
            x[85:95, 0:15, :] = 0

            # normalize the grass
            GREEN_MIN = (100, 200, 100)
            GREEN_MAX = (102, 229, 102)

            mask = cv2.inRange(x, GREEN_MIN, GREEN_MAX)

            # mask out pitch black
            mask_black = cv2.inRange(x, (0, 0, 0), (1, 1, 1))

            # set masked pixels to white
            # x[mask == 255] = 255
            x[mask_black == 255] = 255

            # flip the image if needed
            if self.flipped:
                x = np.flip(x, 1)

            x = transform(x)

            return x

        return preprocess

    def _make_postprocess(self) -> Tuple[int, Callable]:
        num_actions = 2

        def postprocess(action):
            mapped_action = torch.zeros(3)

            # flip steering if flipped
            mapped_action[0] = action[0] if not self.flipped else -action[0]

            # map [-1, 1] throttle to gas and brakes
            mapped_action[1] = max(0, action[1])
            mapped_action[2] = max(0, -action[1])

            return mapped_action.numpy().clip(-1, 1)

        return num_actions, postprocess

    def save(self, folder, episode):
        console.log(f"Saving model at episode {episode} to {folder}")
        torch.save(self.net.state_dict(), path.join(folder, f"ac_{episode}.pth"))

    def load(self, folder, episode):
        console.log(f"Loading checkpoint from episode {episode}...")
        self.net.load_state_dict(torch.load(path.join(folder, f"ac_{episode}.pth")))
        self.n_episodes = episode
