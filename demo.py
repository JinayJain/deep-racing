import toml
from argparse import ArgumentParser
from os.path import join

from games.carracing import RacingNet, CarRacing
from ppo import PPO

CONFIG_FILE = "config.toml"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = toml.load(f)

    return config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt")
    parser.add_argument("--num_steps", type=int, default=100_000)
    parser.add_argument("--delay_ms", type=int, default=10)

    return parser.parse_args()


def main():
    cfg = load_config()
    args = parse_args()

    env = CarRacing(frame_skip=0, frame_stack=4,)
    net = RacingNet(env.observation_space.shape, env.action_space.shape)

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

    ppo.load(args.ckpt)

    for i in range(args.num_steps):
        ppo.collect_trajectory(1, delay_ms=args.delay_ms)

    env.close()


if __name__ == "__main__":
    main()
