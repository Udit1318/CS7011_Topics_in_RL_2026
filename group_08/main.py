import gymnasium as gym
import safety_gymnasium
import torch
import numpy as np
import random
import csv
import pickle
import os
import multiprocessing
import argparse
from agent import FOCOPSAgent
from buffer import RolloutBuffer

# Fully explicit configurations for each environment
ENV_CONFIGS = {
    'Walker2d-v5': {
        'epochs': 2500,
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
        'cost_threshold': 81.89, 
        'cost_key': 'x_velocity', 
        'abs_cost': True
    },
    'HalfCheetah-v4': {
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
    },

    # --- Safety Gym Environments ---
    'SafetyCarGoal1-v0': {
        'epochs': 200,
        'batch_size': 50000,
        'minibatch_size': 1000,
        'optim_epochs': 10,
        'max_ep_len': 1000,
        'gamma': 0.995,
        'gae_lam': 0.95,
        'lr_pi': 3e-4,
        'lr_v': 3e-4,
        'lr_nu': 0.01,
        'temperature_lam': 1.0,
        'trust_region_delta': 0.04,
        'initial_nu': 0.0,
        'nu_max': 2.0,
        'cost_threshold': 7.0, 
        'cost_key': 'cost', 
        'abs_cost': False
    },
    'SafetyCarPush1-v0': {
            'epochs': 200,
            'batch_size': 50000,
            'minibatch_size': 1000,
            'optim_epochs': 10,
            'max_ep_len': 1000,
            'gamma': 0.995,
            'gae_lam': 0.95,
            'lr_pi': 3e-4,
            'lr_v': 3e-4,
            'lr_nu': 0.01,
            'temperature_lam': 1.0,
            'trust_region_delta': 0.04,
            'initial_nu': 0.0,
            'nu_max': 2.0,
            'cost_threshold': 5,
            'cost_key': 'cost', 
            'abs_cost': False
    }
}

def get_cost(info, env_config):
    key = env_config.get('cost_key', 'cost')
    raw_cost = info.get(key, 0.0)
    
    if env_config.get('abs_cost', False):
        return abs(raw_cost)
        
    return float(raw_cost)

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def run_single_seed(seed, env_name, config, is_last_seed):
    print(f"\n{'='*40}\nStarting Training for Env: {env_name} | Seed: {seed}\n{'='*40}")
    
    set_seeds(seed)
    
    seed_dir = os.path.join(env_name, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    csv_filename = os.path.join(seed_dir, f'training_log_seed_{seed}.csv')
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Seed', 'Epoch', 'Dual_Variable_nu', 'Average_Discounted_Cost', 'Average_Returns'])
        
        # Make the environment
        if env_name.startswith('Safety'):
            env = safety_gymnasium.make(env_name)
            env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        else:
            env = gym.make(env_name)

        if isinstance(env.observation_space, gym.spaces.Dict) or 'Franka' in env_name:
            env = gym.wrappers.FlattenObservation(env)
        # Normalize Observations
        env = gym.wrappers.NormalizeObservation(env)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        agent = FOCOPSAgent(obs_dim, act_dim, config)
        buffer = RolloutBuffer(obs_dim, act_dim, config['batch_size'], config['gamma'], config['gae_lam'])

        obs, _ = env.reset(seed=seed) # Initialize environment and get initial observation
        ep_ret, ep_cost, ep_len = 0, 0, 0
        ep_discounted_cost = 0

        for epoch in range(config['epochs']):
            completed_ep_costs = [] 
            completed_ep_rets = []
            completed_ep_discounted_costs = []
            
            for t in range(config['batch_size']): # 2048 steps per epoch
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                
                with torch.no_grad():
                    dist = agent.pi(obs_tensor) # Get action distribution from current policy
                    act = dist.sample() # Sample action from the distribution
                    logp = dist.log_prob(act).sum(axis=-1) # Get ∑_i log π(a_i|s) from log π(a|s) from 𝜋(·|s)
                    val_r = agent.v_reward(obs_tensor) # Get V_r(s) from the reward critic
                    val_c = agent.v_cost(obs_tensor) # Get V_c(s) from the cost critic

                act_np = act.numpy()
                next_obs, rew, terminated, truncated, info = env.step(act_np)
                
                cost = get_cost(info, config)
                ep_discounted_cost += (config['gamma'] ** ep_len) * cost
                buffer.store(obs, act_np, rew, cost, val_r.item(), val_c.item(), logp.item()) # Store transition in buffer with V_r(s) and V_c(s) for GAE calculation
                
                obs = next_obs # Get next observation
                ep_ret += rew
                ep_cost += cost
                ep_len += 1
                
                timeout = ep_len == config['max_ep_len']
                terminal = terminated or truncated or timeout
                epoch_ended = t == config['batch_size'] - 1

                if terminal or epoch_ended:
                    if terminal:
                        completed_ep_costs.append(ep_cost)
                        completed_ep_rets.append(ep_ret)
                        completed_ep_discounted_costs.append(ep_discounted_cost)
                    
                    if timeout or epoch_ended:
                        with torch.no_grad(): # Gradients can be disabled for inference
                            last_val_r = agent.v_reward(torch.as_tensor(obs, dtype=torch.float32)).item() # Get V_r(s') for the last state
                            last_val_c = agent.v_cost(torch.as_tensor(obs, dtype=torch.float32)).item() # Get V_c(s') for the last state
                    else:
                        last_val_r, last_val_c = 0, 0
                    
                    buffer.finish_path(last_val_r, last_val_c) # Compute GAE and Discounted Returns and Costs for the path and reset pointer
                    obs, _ = env.reset()
                    ep_ret, ep_cost, ep_len = 0, 0, 0
                    ep_discounted_cost = 0

            data = buffer.get() # Return all transitions collected in this epoch

            # Compute mean discounted cost across completed episodes in the epoch for updating nu
            if completed_ep_discounted_costs:
                J_c_hat = np.mean(completed_ep_discounted_costs)
            else:
                J_c_hat = ep_discounted_cost
            
            # Update policy, reward critic, cost critic, and dual variable nu using the collected data from the epoch
            agent.update(data, J_c_hat)
            
            avg_ep_ret = np.mean(completed_ep_rets) if completed_ep_rets else ep_ret
            
            print(f"Env {env_name} | Seed {seed} | Epoch {epoch+1}/{config['epochs']} | nu: {agent.nu:.4f} | Cost: {J_c_hat:.2f} | Reward: {avg_ep_ret:.2f}")

            # Log to specific seed's CSV
            csv_writer.writerow([seed, epoch + 1, agent.nu, J_c_hat, avg_ep_ret])
            csv_file.flush()

            # --- Checkpoint Saving Logic ---
            is_mid_training = (epoch == (config['epochs'] // 2) - 1)
            is_end_training = (epoch == config['epochs'] - 1)

            # If it's the end of training OR (it's the last seed AND we are at the midpoint)
            if is_end_training or (is_last_seed and is_mid_training):
                suffix = "mid" if is_mid_training else "end"
                
                torch.save(agent.pi.state_dict(), os.path.join(seed_dir, f'policy_seed_{seed}_{suffix}.pth'))
                with open(os.path.join(seed_dir, f'obs_rms_seed_{seed}_{suffix}.pkl'), 'wb') as f:
                    pickle.dump(env.obs_rms, f)

def main():
    parser = argparse.ArgumentParser(description='Train FOCOPS on different environments.')
    parser.add_argument('--env', type=str, default='Walker2d-v5', help='Name of the environment')
    args = parser.parse_args()

    if args.env not in ENV_CONFIGS:
        raise ValueError(f"Environment {args.env} not found in ENV_CONFIGS.")

    config = ENV_CONFIGS[args.env]
    config['env_name'] = args.env

    # Hardcoded seeds (removed from argparse)
    seeds = [42, 101, 777, 97, 88]

    print(f"Launching {len(seeds)} parallel processes for {args.env}...")
    
    # Pack arguments for starmap: (seed, env_name, config, is_last_seed)
    args_list = [(seed, args.env, config, seed == seeds[-1]) for seed in seeds]
    
    # Run seeds in parallel using multiprocessing Pool
    with multiprocessing.Pool(processes=len(seeds)) as pool:
        pool.starmap(run_single_seed, args_list)

if __name__ == '__main__':
    main()