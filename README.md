# Self-Driving Racecar with Proximal Policy Optimization

Solving the OpenAI Gym [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0) environment using Proximal Policy Optimization.

## Demo

![Video Demo](extra/demo.gif)

## Results

After 5000 training steps, the agent achieves a mean score of 909.48±10.30 over 100 episodes. Results are computed using the `demo.py` script.

## Implementation Details

- A convolutional neural network to jointly approximate the value function and the policy.
- Optimization is performed using [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).
- Policy network outputs parameters to a Beta distribution, [which is better for bounded continuous action spaces](https://proceedings.mlr.press/v70/chou17a/chou17a.pdf).
- A series of 4 frames are concatenated to form the input to the network, with frame skipping optionally applied.
