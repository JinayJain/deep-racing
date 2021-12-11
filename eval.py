from time import sleep
from itertools import count

import gym
import torch
import yaml

from dqn import DQN
from main import make_action_space, preprocess


def main():
    env = gym.make('CarRacing-v0')

    state = preprocess(env.reset())

    with open('config.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    action_space = make_action_space(cfg["actions"])

    dqn = DQN(
        state.shape,
        len(action_space)
    )

    dqn.load('ckpt/700.pt')

    for t in count():
        if t % 3 == 0:
            action, q_value = dqn.get_action(state, use_epsilon=False)

        screen, reward, done, _ = env.step(action_space[action])

        state = preprocess(screen)

        if done:
            env.reset()

        env.render()

        sleep(0.01)


if __name__ == '__main__':
    main()
