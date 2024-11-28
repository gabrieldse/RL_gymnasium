import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import argparse
from stable_baselines3 import PPO, DDPG
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO or DDPG model on a specified environment.")
    parser.add_argument('--model', type=str, required=True, help="Path to the model file (.zip)")
    return parser.parse_args()


def separar_nome():
    args = parse_args()
    arquivo = args.model
    base_name = os.path.basename(arquivo)
    partes = base_name.split('_')
    algo = partes[0]
    parsed = partes[1].split('-')
    env = parsed[0]
    episodio = parsed[2]
    print(f"Episodio = {type(env)}")
    episodio = int(episodio)
    if env.lower() == "Pendulum" or env.lower() == "pendulum":
        env = "Pendulum-v1"
    else:
        env = "MountainCarContinuous-v0"
    return algo, env, episodio, arquivo


def main():
    algo, env, episodes, arq = separar_nome()

    # print(f"Algorithm: {algo}")
    # print(f"Environment: {env}")
    # print(f"Episode: {episodes}")
    # print(f"Arquivo: {arq}")

    # Initialize environment
    eval_env = gym.make(env)
    seed = 42
    obs, info = eval_env.reset(seed=seed)

    # Load the appropriate model
    if algo == "ddpg":
        model = DDPG.load(arq)
        TIMESTEPS_PER_EPISODE = 200
    else:
        model = PPO.load(arq)
        TIMESTEPS_PER_EPISODE = 999

    # Run the environment with the model
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

    # Plot Rewards
    plt.figure(num=1, figsize=(10, 6))
    plt.plot(episode_rewards, label="Récompense par épisode", color="blue")
    plt.plot(cumulative_rewards, label="Récompense accumulé", color="orange")
    plt.xlabel("Épisodes")
    plt.ylabel("Récompense")
    plt.title(f"Récompense pour {episodes} épisodes")
    plt.legend()
    plt.grid()
    # Add training time text below the plot
    plt.subplots_adjust(bottom=0.2)  # Adjust space for the text
    plt.figtext(0.5, 0.05, f"Récompense final de l'entraînement : {total_reward:.2f}", ha="center", fontsize=10, wrap=True)

    # Plot Observations
    if env == "Pendulum-v1":
        cos_theta = [obs[0] for episode in observations for obs in episode]
        sin_theta = [obs[1] for episode in observations for obs in episode]
        angular_velocity = [obs[2] for episode in observations for obs in episode]

        plt.figure(num=2, figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.arctan2(sin_theta, cos_theta)*180/np.pi, label="Angle Theta", color="green")
        plt.ylabel("Theta (degré)")
        plt.title(f"Trajectoire du Pendulum après {episodes} épisodes")
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(angular_velocity, label="Adngular Velocity", color="red")
        plt.xlabel("Timesteps")
        plt.ylabel("Vitesse angulaire (m/s)")
        plt.legend()
        plt.grid()

    else:
        position = [obs[0] for episode in observations for obs in episode]
        velocity = [obs[1] for episode in observations for obs in episode]

        plt.figure(num=2, figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(position, label="Position", color="purple")
        plt.ylabel("Position (m)")
        plt.title(f"Trajectoire du MountainCar après {episodes} épisodes")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(velocity, label="Vitesse", color="pink")
        plt.xlabel("Timesteps")
        plt.ylabel("Vitesse (m/s)")
        plt.grid()
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
