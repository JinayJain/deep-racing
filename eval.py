from time import sleep
from itertools import count

import gym
import torch

from dqn import DQN
from main import preprocess


def main():
    env = gym.make('CarRacing-v0')

    state = preprocess(env.reset())

    dqn = DQN(
        state.shape,
        len(ACTIONS)
    )

    dqn.load('ckpt/700.pt')

    for t in count():
        if t % 3 == 0:
            action, q_value = dqn.get_action(state, use_epsilon=False)

        screen, reward, done, _ = env.step(ACTIONS[action][1])

        state = preprocess(screen)

        if done:
            env.reset()

        env.render()

        sleep(0.01)


if __name__ == '__main__':
    main()
