import gym
from ppo import PPO


def main():
    env = gym.make("CarRacing-v0")

    ppo = PPO(env)
    ppo.train()

    env.close()


if __name__ == "__main__":
    main()
