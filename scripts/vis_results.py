import torch
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from os.path import join

from ppo import PPO
from games.carracing import CarRacing, RacingNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env = CarRacing(frame_skip=1, frame_stack=4,)
    net = RacingNet(env.observation_space.shape, env.action_space.shape).to(device)

    ppo = PPO(env, net)
    ppo.load("ckpt/net_final.pth")

    # Intercept last conv layer output
    activations = None

    def set_activations(module, input, output):
        nonlocal activations
        activations = input[0]

    relu_layer = net.conv[-2]
    relu_layer.register_forward_hook(set_activations)

    for t in range(1000):
        ppo.collect_trajectory(1)

        rgb = env.render(mode="rgb_array")

        global_avg = torch.mean(activations, dim=[0, 2, 3])

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= global_avg[i]

        heatmap = activations.mean([0, 1])
        heatmap /= heatmap.max()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)

        heatmap = cv.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)

        img = (0.7 * rgb + 0.3 * heatmap).astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        cv.imshow("img", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
