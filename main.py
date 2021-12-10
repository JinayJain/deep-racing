import gym
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as T
import yaml
import matplotlib.pyplot as plt

from itertools import count
from os import path
import time

from ddpg import DDPG
from util.plotter import Plotter


CONFIG_FILE = "config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess(img):
    # Normalize according to the pre-trained model (https://pytorch.org/vision/stable/models.html)
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    downsize = T.Resize((64, 64))
    grayscale = T.Grayscale()

    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    # img = grayscale(img)
    img /= 255.0
    img = downsize(img)

    return img


def main():
    cfg = load_config()

    # Seed RNG
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    env = gym.make("CarRacing-v0")
    env.reset()

    print(env.action_space)

    ddpg = DDPG(
        actor_lr=cfg["actor_lr"],
        critic_lr=cfg["critic_lr"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        replay_buffer_size=cfg["replay_buffer_size"],
        batch_size=cfg["batch_size"],
        device=device,
    )

    noise = cfg["noise"]

    plot = cfg["plot"]
    plot_interval = cfg["plot_interval"]

    all_episode_plt = Plotter("All Episodes", "Episode", "Value", plot=plot)
    episode_plt = Plotter("Within Episode", "Step",
                          "Value", update_interval=plot_interval, plot=plot)
    loss_plt = Plotter("Loss", "Step", "Loss",
                       update_interval=plot_interval, plot=plot)
    noise_plt = Plotter("Noise", "Episode", "Noise", plot=plot)

    for ep in range(cfg["num_episodes"]):
        state = preprocess(env.reset()).unsqueeze(0).to(device)

        last_reward_step = 0
        total_reward = 0

        episode_plt.reset()
        loss_plt.reset()

        for t in count():
            start = time.time()
            # Create a new transition to store in the replay buffer
            with torch.no_grad():
                # Sample an action from the policy (noise is added to ensure exploration)
                action = ddpg.actor(state)
                action += torch.randn(action.shape).to(device) * noise
                action.clamp_(min=0, max=1)

                # Ask the critic for the Q-value estimate of the current state and action
                q_value = ddpg.critic(state, action)

                env_action = action.squeeze(0).cpu().numpy(
                ) * np.array([2.0, 1.0, 1.0]) + np.array([-1.0, 0.0, 0.0])

                print(env_action)
                print(action)

                screen, reward, done, _ = env.step(env_action)

                next_state = preprocess(screen).unsqueeze(0).to(device)
                total_reward += reward

                target_next_q = ddpg.target_critic(state, action)

                if not done:
                    td_error = (reward + cfg["gamma"]
                                * target_next_q - q_value).item()
                else:
                    td_error = reward - q_value.item()

                if reward > 0:
                    last_reward_step = t

                if done or (t - last_reward_step) > cfg["max_steps_without_reward"]:
                    ddpg.push(state.cpu(), action.cpu(), reward,
                              None, td_error)
                    break

                ddpg.push(
                    state.cpu(), action.cpu(), reward, next_state.cpu(), td_error
                )

            if cfg["render"]:
                env.render()

            # Train on a sampled batch from the replay buffer
            actor_loss, critic_loss = ddpg.train_batch()

            loss_plt.append(t, actor_loss, "Actor Loss")
            loss_plt.append(t, critic_loss, "Critic Loss")

            state = next_state

            episode_plt.append(t, q_value.item(), "Q-Value")
            episode_plt.append(t, target_next_q.item(), "Target Next Q-Value")
            episode_plt.append(t, total_reward, "Total Reward")
            episode_plt.append(t, td_error, "TD Error")

            end = time.time()

            print(
                f"Episode: {ep} | Step: {t} | Reward: {reward} | Time: {end - start}")

        print(
            f"Episode {ep} finished after {t} timesteps with total reward {total_reward}"
        )

        noise = max(cfg["noise_decay"] * noise, cfg["noise_min"])

        all_episode_plt.append(ep, total_reward, "Total Reward")
        noise_plt.append(ep, noise, "Noise")

        if ep % cfg["save_every"] == 0:
            ddpg.save(path.join(cfg["save_dir"], f"{ep}"))

    env.close()


if __name__ == "__main__":
    main()
