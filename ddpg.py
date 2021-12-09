import torch
from torch import optim, nn

from net import Actor, Critic
from util.memory import ReplayBuffer


class DDPG:
    def __init__(
        self,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.001,
        replay_buffer_size=1e6,
        batch_size=32,
        device=None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.actor = Actor().to(self.device)
        self.target_actor = Actor().to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic().to(self.device)
        self.target_critic = Critic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_critic.eval()
        self.criterion = nn.MSELoss()

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayBuffer(replay_buffer_size)

    def push(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def train_batch(self):
        if len(self.memory) < self.batch_size:
            return 0, 0

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.cat(actions, dim=0).to(self.device)
        rewards = torch.tensor(rewards).to(self.device).unsqueeze(1)

        # Handle terminal states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            dtype=torch.bool,
            device=self.device,
        )

        next_states = torch.cat([s for s in next_states if s is not None]).to(
            self.device
        )

        self.critic_opt.zero_grad()

        # Compute predicted Q-values
        q_pred = self.critic(states, actions)

        # Compute target Q-values
        q_target = rewards
        q_target[non_final_mask] += self.gamma * self.target_critic(
            next_states, self.target_actor(next_states)
        )

        # Optimize critic by minimizing the MSE between preds and targets
        loss = self.criterion(q_pred, q_target)
        loss.backward()
        self.critic_opt.step()

        # Optimize actor using the policy gradient update
        self.actor_opt.zero_grad()
        actor_loss = -self.critic(
            states, self.actor(states)
        ).mean()  # this should be maximized
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update_targets()

        return actor_loss.item(), loss.item()

    def soft_update_targets(self):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

        for target_param, param in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, path):
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "_critic.pth"))
