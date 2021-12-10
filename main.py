import gym
import numpy as np
import torch
import torchvision.transforms as T
import yaml

import time
from itertools import count

from dqn import DQN

# from util.plotter import Plotter


CONFIG_FILE = "config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


def preprocess(img):
    # Normalize according to the pre-trained model (https://pytorch.org/vision/stable/models.html)
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    downsize = T.Resize((64, 64))
    grayscale = T.Grayscale()

    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = grayscale(img)
    img /= 255.0
    img = downsize(img)

    return img


def main():
    cfg = load_config()

    # Seed RNG
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    env = gym.make("CarRacing-v0")

    action_types = np.array(
        [
            [-1.0, 0.0, 0.0],  # turn left
            [1.0, 0.0, 0.0],  # turn right
            [0.0, 1.0, 0.0],  # accelerate
            [0.0, 0.0, 1.0],  # brake
            [0.0, 0.0, 0.0],  # no-op
        ]
    )

    sample_state = preprocess(env.reset())

    dqn = DQN(
        sample_state.shape,
        action_types.shape[0],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon=cfg["epsilon"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay=cfg["epsilon_decay"],
        memory_size=cfg["memory_size"],
        memory_alpha=cfg["memory_alpha"],
        memory_beta=cfg["memory_beta"],
    )

    for ep in range(cfg["num_episodes"]):
        state = preprocess(env.reset())

        for t in count():
            start = time.time()
            with torch.no_grad():
                action, q_value = dqn.get_action(state)

                screen, reward, done, _ = env.step(action_types[action])
                next_state = preprocess(screen)

                if not done:
                    error = dqn.td_error(q_value[action], reward, next_state)
                    dqn.remember(state, action, reward, next_state, error)
                else:
                    error = (q_value[action] - reward).item()

            loss = dqn.train(cfg["batch_size"])

            env.render()

            end = time.time()

            print(f"{end - start}")

        dqn.decay_epsilon()

    env.close()


if __name__ == "__main__":
    main()
