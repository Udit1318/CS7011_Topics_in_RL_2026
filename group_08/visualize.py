import gymnasium as gym
import safety_gymnasium
import torch
import imageio
import numpy as np
import pickle
import os
import glob
import argparse
from agent import FOCOPSAgent
from main import ENV_CONFIGS

def record_videos(env_name):
    print(f"Searching for policies in directory '{env_name}'...")
    
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Environment {env_name} not found in ENV_CONFIGS.")
    
    config = {
        'epochs': 2000,
        'batch_size': 2048,
        'minibatch_size': 64,
        'optim_epochs': 10,
        'max_ep_len': 1000,
        'gamma': 0.99,
        'gae_lam': 0.95,
        'lr_pi': 3e-4,
        'lr_v': 3e-4,
        'lr_nu': 0.01,
        'temperature_lam': 1.5,
        'trust_region_delta': 0.02,
        'initial_nu': 0.0,
        'nu_max': 2.0,
        'cost_threshold': 151.99, 
        'cost_key': 'x_velocity', 
        'abs_cost': True
    }
    
    # Recursively find all policy .pth files in the env_name directory
    policy_pattern = os.path.join(env_name, "**", "*.pth")
    policy_files = glob.glob(policy_pattern, recursive=True)
    
    if not policy_files:
        print(f"No policy files found in '{env_name}'. Make sure training has completed.")
        return
        
    print(f"Found {len(policy_files)} policies. Generating videos...")
    
    # Initialize the environment and agent once
    if env_name.startswith('Safety'):
        env = safety_gymnasium.make(env_name, render_mode="rgb_array",width=1280, 
            height=720)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    else:
        env = gym.make(env_name, render_mode="rgb_array",width=1280, 
            height=720)
    env = gym.wrappers.NormalizeObservation(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = FOCOPSAgent(obs_dim, act_dim, config)
    
    for policy_path in policy_files:
        # Extract directory and filename to keep outputs in the same folder
        policy_dir = os.path.dirname(policy_path)
        policy_filename = os.path.basename(policy_path)
        
        # Deduce corresponding normalizer (.pkl) and video (.mp4) filenames
        # Example: policy_seed_88_mid.pth -> obs_rms_seed_88_mid.pkl
        rms_filename = policy_filename.replace("policy_", "obs_rms_").replace(".pth", ".pkl")
        rms_path = os.path.join(policy_dir, rms_filename)
        
        video_filename = policy_filename.replace("policy_", "simulation_").replace(".pth", ".mp4")
        video_path = os.path.join(policy_dir, video_filename)
        
        print(f"\n--- Processing: {policy_filename} ---")
        
        # Load Observation Normalizer
        try:
            with open(rms_path, 'rb') as f:
                env.obs_rms = pickle.load(f)
            print(f"Loaded normalizer: {rms_filename}")
        except FileNotFoundError:
            print(f"Could not find matching normalizer {rms_path}. Skipping.")
            continue

        # Load Policy Weights
        try:
            agent.pi.load_state_dict(torch.load(policy_path))
            print(f"Loaded policy: {policy_filename}")
        except Exception as e:
            print(f"Error loading policy {policy_path}: {e}")
            continue

        frames = []
        obs, _ = env.reset()
        
        print(f"Recording frames for {video_filename}...")
        for step in range(config['max_ep_len']):
            frames.append(env.render())
            
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                act = agent.pi(obs_tensor).loc.numpy() 
                
            obs, reward, terminated, truncated, info = env.step(act)
            
            if terminated or truncated:
                print(f"Episode finished early at step {step}.")
                break

        print(f"Saving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=30)
        print("Done.")
        
    env.close()
    print("\nAll videos generated successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate videos for all trained FOCOPS agents in an environment directory.")
    parser.add_argument('--env', type=str, required=True, help="Name of the environment (e.g., Walker2d-v5)")
    
    args = parser.parse_args()
    record_videos(args.env)