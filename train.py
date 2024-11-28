import gymnasium as gym
import numpy as np
import argparse
import os
from datetime import datetime
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure
from gymnasium.spaces import Box

# Custom environment with modified rewards
class ModifiedMountainCarEnv(gym.Env):
    def __init__(self):
        super(ModifiedMountainCarEnv, self).__init__()
        self.env = gym.make("MountainCarContinuous-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Reward shaping parameters
        self.success_bonus = 100.0
        self.proximity_scale = 10.0
        self.velocity_scale = 5.0
        self.direction_penalty = -1.0
        self.goal_position = 0.45
        self.previous_position = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_position = obs[0]
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        position, velocity = obs[0], obs[1]

        # Reward modifications
        reward += max(0, (position - self.previous_position) * self.proximity_scale)
        if velocity > 0:
            reward += velocity * self.velocity_scale
        if position < self.previous_position:
            reward += self.direction_penalty
        if done and position >= self.goal_position:
            reward += self.success_bonus

        self.previous_position = position
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()

# Helper function to log training duration
def log_times(start_time, end_time, filename):
    duration = (end_time - start_time).total_seconds()
    new_filename = filename.replace('.zip', f"-{duration:.02f}-s.zip")
    if os.path.exists(filename):
        os.rename(filename, new_filename)
        print(f"Renamed '{filename}' to '{new_filename}'")
    else:
        print(f"File '{filename}' does not exist. Cannot rename.")

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO or DDPG on a custom environment.")
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'ddpg'], help="Choose the RL algorithm (PPO or DDPG)")
    parser.add_argument('--env', type=str, required=True, help="Gymnasium environment name (e.g., 'Pendulum-v1', 'MountainCarContinuous-v0', 'ModifiedMountainCarEnv-v0')")
    parser.add_argument('--save_path', type=str, default="./data", help="Path to save the model and checkpoints")
    return parser.parse_args()

# Main function to train the agent
def main():
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Check if the environment is the modified one and initialize accordingly
    if args.env == "ModifiedMountainCarEnv-v0":
        env = ModifiedMountainCarEnv()  # Initialize custom environment
    else:
        env = gym.make(args.env, render_mode="rgb_array")

    # Define the timesteps per episode based on environment
    if args.env == "Pendulum-v1":
        timesteps_per_episode = 200
    elif args.env in ["MountainCarContinuous-v0", "ModifiedMountainCarEnv-v0"]:
        timesteps_per_episode = 999

    # Setup the model (DDPG or PPO)
    if args.algo == "ddpg":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    else:  # PPO
        model = PPO("MlpPolicy", env, verbose=1)

    # Set up TensorBoard logging
    log_dir = os.path.join(args.save_path, "tensorboard")
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    start_time = datetime.now()
    checkpoint_interval = 10  # Save model every 10 episodes
    total_episodes = 1000  # Modify this for longer training

    # Training loop
    try:
        for episode in range(1, total_episodes + 1):
            print(f"Starting training for episode {episode}...")
            model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)

            if episode % checkpoint_interval == 0:
                checkpoint_file = os.path.join(args.save_path, f"{args.algo}_{args.env}-{episode}-ep.zip")
                model.save(checkpoint_file)
                print(f"Checkpoint saved at {checkpoint_file}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    final_file = os.path.join(args.save_path, f"{args.algo}_{args.env}-final.zip")
    model.save(final_file)
    end_time = datetime.now()
    log_times(start_time, end_time, final_file)

    print(f"Final model saved at {final_file}. Training complete.")

if __name__ == "__main__":
    main()
