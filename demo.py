import gym

from ppo import PPO

CHECKPOINT_DIR = "ckpt"


def main():
    env = gym.make("CarRacing-v0")

    ppo = PPO(env)
    ppo.load(CHECKPOINT_DIR, 1100)
    ppo.sample_trajectories()

    env.close()


if __name__ == "__main__":
    main()

