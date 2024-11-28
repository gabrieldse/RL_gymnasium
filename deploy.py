import gymnasium as gym
import argparse
from stable_baselines3 import PPO, DDPG

def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO or DDPG model on a specified environment.")
    parser.add_argument('--algo', type=str, required=True, choices=['ppo', 'ddpg'], help="Choose the RL algorithm (PPO or DDPG)")
    parser.add_argument('--env', type=str, required=True, help="Gymnasium environment name (e.g., 'Pendulum-v1', 'MountainCarContinuous-v0')")
    parser.add_argument('--model', type=str, required=True, help="Path to the model file (.zip)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Initialize environment
    env = gym.make(args.env, render_mode="human")

    # Load the appropriate model
    if args.algo == "ddpg":
        model = DDPG.load(args.model)
    else:  # PPO
        model = PPO.load(args.model)

    obs, _ = env.reset()
    
    # Run the environment with the model
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()

        if dones or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
