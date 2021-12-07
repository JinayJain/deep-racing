import gym
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as T
import yaml

from actor import Actor
from critic import Critic

CONFIG = "config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_backbone():
    bb = efficientnet_b0(pretrained=True)
    bb.classifier = nn.Identity()
    bb.eval()

    return bb


def preprocess(img):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img /= 255.0
    img = normalize(img)

    return img


def main():
    cfg = load_config()

    effnet = make_backbone().to(device)

    env = gym.make("CarRacing-v0")
    env.reset()

    actor = Actor()
    critic = Critic()

    for ep in range(cfg["num_episodes"]):
        screen = env.reset()
        done = False

        while not done:
            screen = preprocess(screen)

            with torch.no_grad():
                obs = effnet(screen.unsqueeze(0).to(device))

            print(obs)

            screen, reward, done, _ = env.step(env.action_space.sample())

            env.render()

    env.close()


if __name__ == '__main__':
    main()
