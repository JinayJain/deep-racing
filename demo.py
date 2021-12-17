import gym

from ppo import PPO

CHECKPOINT_DIR = "ckpt"


def main():
    env = gym.make("BipedalWalker-v3")

    ppo = PPO(
        env,
        variance=0.1,
        epoch_timesteps=1,
        max_episode_timesteps=256,
        render_interval=1,
    )

    start_max_timesteps = 128
    end_max_timesteps = 2048

    episode_nums = range(50, 901, 50)

    # linearly scale the max_timesteps to the end_max_timesteps
    for episode_num in episode_nums:
        ppo.max_episode_timesteps = int(
            start_max_timesteps
            + ((episode_num - 50) / (900 - 50))
            * (end_max_timesteps - start_max_timesteps)
        )
        ppo.load(CHECKPOINT_DIR, episode_num)
        ppo.sample_trajectories()

        print(f"Showing results from episode {episode_num}")

    env.close()


if __name__ == "__main__":
    main()

