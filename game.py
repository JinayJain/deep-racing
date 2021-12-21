import gym
from gym.spaces import Box
import numpy as np
from collections import deque


class CarRacing(gym.Wrapper):
    def __init__(self, frame_skip=0, frame_stack=4):
        self.env = gym.make("CarRacing-v0")
        super().__init__(self.env)

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack

        self.action_space = Box(low=-1, high=1, shape=(2,))
        self.observation_space = Box(low=0, high=1, shape=(frame_stack, 96, 96))

        self.frame_buf = deque(maxlen=frame_stack)

    def preprocess(self, original_action):
        action = np.zeros(3)

        action[0] = original_action[0]

        # Separate acceleration and braking
        action[1] = max(0, original_action[1])
        action[2] = max(0, -original_action[1])

        return action

    def postprocess(self, original_observation):
        # convert to grayscale
        grayscale = np.array([0.299, 0.587, 0.114])
        observation = np.dot(original_observation, grayscale) / 255.0

        return observation

    def get_observation(self):
        return np.array(self.frame_buf)

    def reset(self):
        first_frame = self.postprocess(self.env.reset())

        for _ in range(self.frame_stack):
            self.frame_buf.append(first_frame)

        return self.get_observation()

    def step(self, action):
        action = self.preprocess(action)

        new_frame, reward, done, info = self.env.step(action)

        for _ in range(self.frame_skip):
            new_frame, reward, done, info = self.env.step(action)

        new_frame = self.postprocess(new_frame)
        self.frame_buf.append(new_frame)

        return self.get_observation(), reward, done, info

