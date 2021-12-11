import gym
import numpy as np
import torch
import torchvision.transforms as T
import yaml

import time
from itertools import count
from os import path

from dqn import DQN

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
    # img = downsize(img)

    return img


def make_action_space(action_levels):
    actions = []

    for i in range(len(action_levels["turn"])):
        for j in range(len(action_levels["gas"])):
            for k in range(len(action_levels["brake"])):
                actions.append(
                    [action_levels["turn"][i], action_levels["gas"][j], action_levels["brake"][k]])

    return actions


def main():
    cfg = load_config()

    # Seed RNG
    # np.random.seed(cfg["seed"])
    # torch.manual_seed(cfg["seed"])

    env = gym.make("CarRacing-v0")

    sample_state = preprocess(env.reset())

    action_space = make_action_space(cfg["actions"])

    print(action_space)

    dqn = DQN(
        sample_state.shape,
        len(action_space),
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_steps=cfg["epsilon_steps"],
        memory_size=cfg["memory_size"],
        memory_alpha=cfg["memory_alpha"],
        memory_beta=cfg["memory_beta"],
    )

    loss_plot = Plotter("Loss", "Step", "Loss", update_interval=50)
    episode_plot = Plotter("Within Episode", "Step",
                           "Value", update_interval=50)

    episode_summary_plot = Plotter("Episode Summary", "Episode", "Value")

    global_counter = count()
    no_reward_timeout = cfg["no_reward_timeout"]

    for ep in range(cfg["num_episodes"]):
        state = preprocess(env.reset())

        loss_plot.reset()
        episode_plot.reset()

        last_reward = 0
        total_reward = 0

        start = time.time()
        for t, global_t in zip(count(), global_counter):
            with torch.no_grad():
                action, q_value = dqn.get_action(state)

                screen, reward, done, _ = env.step(action_space[action])

                # Clip reward to -1, 1
                if reward > 0:
                    reward = 5

                next_state = preprocess(screen)

                total_reward += reward

                if reward > 0:
                    last_reward = t

                if done or (t - last_reward) > no_reward_timeout:
                    error = (q_value[action] - reward).item()
                    dqn.remember(state, action, reward, None, error)
                    break

                error = dqn.td_error(q_value[action], reward, next_state)
                dqn.remember(state, action, reward, next_state, error)

                state = next_state

            loss = dqn.train(cfg["batch_size"])
            dqn.step_epsilon()

            loss_plot.append(t, loss, "Loss")
            # episode_plot.append(t, total_reward, "Total Reward")
            episode_plot.append(t, error, "TD Error")

            # plot best Q-value
            episode_plot.append(t, q_value[action].item(), "Q-Value")

            if cfg["render"]:
                env.render()

            if global_t % cfg["target_update_interval"] == 0:
                dqn.update_target()
                print("UPDATING TARGET NETWORK")

        end = time.time()

        print(
            f"Episode {ep} finished | FPS: {t / (end - start):.2f} | Total Reward: {total_reward:.2f}")

        episode_summary_plot.append(ep, total_reward, "Total Reward")
        episode_summary_plot.append(ep, dqn.epsilon, "Epsilon")

        no_reward_timeout = min(
            no_reward_timeout + cfg["no_reward_incr"], cfg["no_reward_max"])

        if ep % cfg["save_every"] == 0:
            dqn.save(path.join(cfg["save_dir"], f"{ep}.pt"))

    env.close()


if __name__ == "__main__":
    main()
