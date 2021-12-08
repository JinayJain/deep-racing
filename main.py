import gym
import numpy as np
import torch
from torch import nn
from torchvision.models import efficientnet_b0
import torchvision.transforms as T
import yaml

from itertools import count

from actor import Actor
from critic import Critic
from memory import ReplayBuffer

CONFIG_FILE = "config.yml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config


def make_backbone():
    bb = efficientnet_b0(pretrained=True)
    bb.classifier = nn.Identity()
    bb.eval()

    return bb


def preprocess(img):
    # Normalize according to the pre-trained model (https://pytorch.org/vision/stable/models.html)
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

    actor = Actor().to(device)
    critic = Critic().to(device)

    memory = ReplayBuffer(cfg["replay_buffer_size"])

    for ep in range(cfg["num_episodes"]):
        with torch.no_grad():
            screen = env.reset()
            state = effnet(preprocess(screen).unsqueeze(0).to(device))

        for t in count():
            # Create a new transition to store in the replay buffer
            with torch.no_grad():
                # Sample an action from the policy (noise is added to ensure exploration)
                action = actor(state).cpu()
                action += torch.randn(action.shape) * cfg["noise_std"]

                screen, reward, done, _ = env.step(action.numpy()[0])

                if done:
                    memory.push(state.cpu(), action, reward, None)
                    break

                next_state = effnet(preprocess(screen).unsqueeze(0).to(device))
                memory.push(state.cpu(), action, reward, next_state.cpu())

            # Train on a sampled batch from the replay buffer

            env.render()

            state = next_state

    env.close()


if __name__ == '__main__':
    main()
