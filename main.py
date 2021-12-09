import gym
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as T
import yaml

from itertools import count
from os import path

from ddpg import DDPG

# from plotter import Plotter

CONFIG_FILE = "config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main():
    cfg = load_config()

    effnet = make_backbone().to(device)

    env = gym.make("CarRacing-v0")
    env.reset()

    ddpg = DDPG(
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        replay_buffer_size=cfg["replay_buffer_size"],
        batch_size=cfg["batch_size"],
        device=device,
    )

    # per_episode_reward = Plotter("Episode Rewards", "Episode", "Reward")
    # episode_reward = Plotter("Reward", "Timestep", "Total Reward")

    noise = cfg["noise"]
    for ep in range(cfg["num_episodes"]):
        # episode_reward.clear()

        with torch.no_grad():
            screen = env.reset()
            state = effnet(preprocess(screen).unsqueeze(0).to(device))

        last_reward_step = 0
        total_reward = 0
        for t in count():
            # Create a new transition to store in the replay buffer
            with torch.no_grad():
                # Sample an action from the policy (noise is added to ensure exploration)
                action = ddpg.actor(state)
                action += torch.randn(action.shape).to(device) * noise

                # Ask the critic for the Q-value estimate of the current state and action
                q_value = ddpg.critic(state, action)
                target_q_value = ddpg.target_critic(state, action)

                action = action.cpu()

                screen, reward, done, _ = env.step(action.numpy()[0])
                total_reward += reward

                # episode_reward.append(t, total_reward, "Total Reward")
                # episode_reward.append(t, q_value.item(), "Q-Value")
                # episode_reward.append(t, target_q_value.item(), "Target Q-Value")

                if reward > 0:
                    last_reward_step = t

                if done or (t - last_reward_step) > cfg["max_steps_without_reward"]:
                    ddpg.push(state.cpu(), action, reward, None)
                    break

                next_state = effnet(preprocess(screen).unsqueeze(0).to(device))

                ddpg.push(state.cpu(), action, reward, next_state.cpu())

            # Train on a sampled batch from the replay buffer
            ddpg.train_batch()

            if cfg["render"]:
                env.render()

            state = next_state

        print(f"Episode {ep} finished after {t} timesteps")

        noise = max(cfg["noise_decay"] * noise, cfg["noise_min"])

        # per_episode_reward.append(ep, total_reward)

        if ep % cfg["save_every"] == 0:
            ddpg.save(path.join(cfg["save_dir"], f"{ep}"))

    env.close()


if __name__ == "__main__":
    main()
