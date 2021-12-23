import torch
import numpy as np
import random
import toml

from game import CarRacing
from net import ActorCritic
from ppo import PPO

CONFIG_FILE = "config.toml"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)

    return config


def main():
    cfg = load_config()

    env = CarRacing(frame_skip=0, frame_stack=4,)
    net = ActorCritic(env.observation_space.shape, env.action_space.shape)

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
    ppo.load(cfg["save_dir"], 5000)
    for i in range(100):
        ppo.collect_trajectory(1000)

    env.close()


if __name__ == "__main__":
    main()
