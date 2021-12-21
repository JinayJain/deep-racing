import gym

from game import CarRacing
from matplotlib import pyplot as plt


def main():
    env = CarRacing(frame_skip=4, frame_stack=4,)

    for _ in range(100):
        env.reset()
        for _ in range(100):
            obs, reward, done, info = env.step(env.action_space.sample())

            # plt.title(_)
            # plt.imshow(obs[0])
            # plt.show()
            # plt.imshow(obs[1])
            # plt.show()
            # plt.imshow(obs[2])
            # plt.show()
            # plt.imshow(obs[3])
            # plt.show()

            env.render()


if __name__ == "__main__":
    main()
