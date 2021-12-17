import gym
from ppo import PPO


def main():
    # env = gym.make("Pendulum-v1")
    # env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("BipedalWalker-v3")

    ppo = PPO(env)
    ppo.train()

    env.close()


if __name__ == "__main__":
    main()

