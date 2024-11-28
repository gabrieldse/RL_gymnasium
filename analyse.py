import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import argparse
from stable_baselines3 import PPO, DDPG
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO or DDPG models on a specified environment.")
    parser.add_argument('--models', type=str, nargs='+', required=True, help="Paths to the model files (.zip)")
    return parser.parse_args()

def parse_name(file):
    base_name = os.path.basename(file)
    parts = base_name.split('_')
    algo = parts[0]
    parsed = parts[1].split('-')
    env = parsed[0]
    episode = parsed[2]
    episode = int(episode)
    if env.lower() == "pendulum":
        env = "Pendulum-v1"
    else:
        env = "MountainCarContinuous-v0"
    return algo, env, episode, file

def run_model(file):
    algo, env, episodes, arq = parse_name(file)

    # Init environment
    eval_env = gym.make(env)
    seed = 42

    if algo == "ddpg":
        model = DDPG.load(arq)
        TIMESTEPS_PER_EPISODE = 200
    else:
        model = PPO.load(arq)
        TIMESTEPS_PER_EPISODE = 999

    episode_rewards = []
    cumulative_rewards = []
    observations = []
    total_reward = 0
    episode_obs = []

    obs, _ = eval_env.reset(seed=seed)
    for _ in range(TIMESTEPS_PER_EPISODE):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        episode_obs.append(obs)
        total_reward += reward
        episode_rewards.append(reward)
        cumulative_rewards.append(total_reward)
        if done or truncated:
            break
    observations.append(np.array(episode_obs))
    eval_env.close()

    return episode_rewards, cumulative_rewards, observations, total_reward, episodes, env

def main():
    args = parse_args()
    models = args.models

    plt.figure(num=1, figsize=(10, 6))
    plt.figure(num=2, figsize=(12, 6))

    for file in models:
        episode_rewards, cumulative_rewards, observations, total_reward, episodes, env = run_model(file)

        # Plot Rewards
        plt.figure(1)
        # plt.plot(episode_rewards, label=f"{file} - Episode Rewards")
        plt.plot(cumulative_rewards, label=f"{file} - Cumulative Rewards")

        # Plot Observations
        if env == "Pendulum-v1":
            cos_theta = [obs[0] for episode in observations for obs in episode]
            sin_theta = [obs[1] for episode in observations for obs in episode]
            angular_velocity = [obs[2] for episode in observations for obs in episode]

            plt.figure(2)
            plt.subplot(2, 1, 1)
            plt.plot(np.arctan2(sin_theta, cos_theta) * 180 / np.pi, label=f"{file} - Theta Angle")
            plt.subplot(2, 1, 2)
            plt.plot(angular_velocity, label=f"{file} - Angular Velocity")
        else:
            position = [obs[0] for episode in observations for obs in episode]
            velocity = [obs[1] for episode in observations for obs in episode]

            plt.figure(2)
            plt.subplot(2, 1, 1)
            plt.plot(position, label=f"{file} - Position")
            plt.subplot(2, 1, 2)
            plt.plot(velocity, label=f"{file} - Velocity")

    # Finalize and Show Plots
    plt.figure(1)
    plt.xlabel("Timesteps")
    plt.ylabel("Rewards")
    plt.title("Rewards for Multiple Models")
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.ylabel("Position / Angle")
    plt.title("Trajectory for Multiple Models")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.xlabel("Timesteps")
    plt.ylabel("Velocity / Angular Velocity")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
