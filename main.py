import gym
import numpy as np
import torch
from torch import nn, optim
from torchvision.models import efficientnet_b0
import torchvision.transforms as T
import yaml
import visdom

from itertools import count

from actor import Actor
from critic import Critic
from memory import ReplayBuffer

CONFIG_FILE = "config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_backbone():
    bb = efficientnet_b0(pretrained=True)
    bb.classifier = nn.Identity()  # remove the final classification layer
    bb.eval()

    return bb


def preprocess(img):
    # Normalize according to the pre-trained model (https://pytorch.org/vision/stable/models.html)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img /= 255.0
    img = normalize(img)

    return img


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def main():
    cfg = load_config()

    effnet = make_backbone().to(device)

    env = gym.make("CarRacing-v0")
    env.reset()

    actor = Actor().to(device)
    critic = Critic().to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=cfg["actor_lr"])
    critic_opt = optim.Adam(critic.parameters(), lr=cfg["critic_lr"])

    # Create target networks
    target_actor = Actor().to(device)
    target_critic = Critic().to(device)

    # Copy the weights from the networks to their target networks
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # Set target networks to evaluation mode
    target_actor.eval()
    target_critic.eval()

    memory = ReplayBuffer(cfg["replay_buffer_size"])

    criterion = nn.MSELoss()

    def train_batch():
        if len(memory) < cfg["batch_size"]:
            return

        batch = memory.sample(cfg["batch_size"])
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states, dim=0).to(device)
        actions = torch.cat(actions, dim=0).to(device)
        rewards = torch.tensor(rewards).to(device).unsqueeze(1)

        # Handle terminal states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, next_states)),
            dtype=torch.bool,
            device=device,
        )

        next_states = torch.cat([s for s in next_states if s is not None]).to(device)

        critic_opt.zero_grad()

        # Compute predicted Q-values
        q_pred = critic(states, actions)

        # Compute target Q-values
        q_target = rewards
        q_target[non_final_mask] += cfg["gamma"] * target_critic(
            next_states, target_actor(next_states)
        )

        # Optimize critic by minimizing the MSE between preds and targets
        loss = criterion(q_pred, q_target)
        loss.backward()
        critic_opt.step()

        # Optimize actor using the policy gradient update
        actor_opt.zero_grad()
        actor_loss = -critic(states, actor(states)).mean()  # this should be maximized
        actor_loss.backward()
        actor_opt.step()

    episode_rewards = []

    # Create episode reward plot
    win_ep_reward = vis.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(title="Episode Rewards", xlabel="Episode", ylabel="Reward",),
    )

    for ep in range(cfg["num_episodes"]):
        with torch.no_grad():
            screen = env.reset()
            state = effnet(preprocess(screen).unsqueeze(0).to(device))

        last_reward_step = 0
        total_reward = 0
        for t in count():
            # Create a new transition to store in the replay buffer
            with torch.no_grad():
                # Sample an action from the policy (noise is added to ensure exploration)
                action = actor(state)
                action += torch.randn(action.shape).to(device) * cfg["noise_std"]

                # Ask the critic for the Q-value estimate of the current state and action
                q_value = critic(state, action)

                action = action.cpu()

                screen, reward, done, _ = env.step(action.numpy()[0])
                total_reward += reward

                if reward > 0:
                    last_reward_step = t

                if done or (t - last_reward_step) > cfg["max_steps_without_reward"]:
                    memory.push(state.cpu(), action, reward, None)
                    break

                next_state = effnet(preprocess(screen).unsqueeze(0).to(device))
                memory.push(state.cpu(), action, reward, next_state.cpu())

            # Train on a sampled batch from the replay buffer
            train_batch()

            # Soft update the target networks
            soft_update(target_actor, actor, cfg["tau"])
            soft_update(target_critic, critic, cfg["tau"])

            if cfg["render"]:
                env.render()

            state = next_state

        print(f"Episode {ep} finished after {t} timesteps")

        episode_rewards.append(total_reward)

        # Update the plot
        vis.line(
            X=np.array([ep]),
            Y=np.array([total_reward]),
            win=win_ep_reward,
            name="Episode Reward",
            update="append",
        )

        if ep % cfg["save_every"] == 0:
            torch.save(actor.state_dict(), f"ckpt/actor_{ep}.pt")
            torch.save(critic.state_dict(), f"ckpt/critic_{ep}.pt")

    env.close()


if __name__ == "__main__":
    main()
