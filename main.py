import gym

from game import CarRacing
from net import ActorCritic
from ppo import PPO


def main():
    env = CarRacing(frame_skip=0, frame_stack=4,)
    net = ActorCritic(env.observation_space.shape, env.action_space.shape)

    ppo = PPO(env, net)
    ppo.train()

    env.close()


if __name__ == "__main__":
    main()
