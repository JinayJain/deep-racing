import torch
import numpy as np
import random
import toml

from games.carracing import RacingNet, CarRacing
from games.bipedal import BipedalNet, BipedalWalker
from ppo import PPO

CONFIG_FILE = "config.toml"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)

    return config


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    cfg = load_config()
    seed(cfg["seed"])

    # env = CarRacing(frame_skip=1, frame_stack=4,)
    # net = RacingNet(env.observation_space.shape, env.action_space.shape)

    env = BipedalWalker()
    net = BipedalNet(env.observation_space.shape, env.action_space.shape)

    ppo = PPO(
        env,
        net,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        gae_lambda=cfg["gae_lambda"],
        clip=cfg["clip"],
        value_coef=cfg["value_coef"],
        entropy_coef=cfg["entropy_coef"],
        epochs_per_step=cfg["epochs_per_step"],
        num_steps=cfg["num_steps"],
        horizon=cfg["horizon"],
        save_dir=cfg["save_dir"],
        save_interval=cfg["save_interval"],
    )
    ppo.train()

    env.close()


if __name__ == "__main__":
    main()
