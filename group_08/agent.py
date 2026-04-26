import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import Actor, Critic
from torch.distributions import Normal

class FOCOPSAgent:
    def __init__(self, obs_dim, act_dim, config):
        self.config = config
        
        self.pi = Actor(obs_dim, act_dim)
        self.v_reward = Critic(obs_dim)
        self.v_cost = Critic(obs_dim)
        
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=config['lr_pi'])
        self.v_r_optimizer = optim.Adam(self.v_reward.parameters(), lr=config['lr_v'])
        self.v_c_optimizer = optim.Adam(self.v_cost.parameters(), lr=config['lr_v'])
        
        self.nu = config['initial_nu']

    def update(self, data, J_c_hat):
        obs = data['obs']
        acts = data['act']
        adv_r = data['adv_r']
        adv_c = data['adv_c']
        ret_r = data['ret_r']
        ret_c = data['ret_c']
        old_logp = data['logp']

        with torch.no_grad():
            old_dist = self.pi(obs) # Old distribution for KL calculation

        # Calculate batch cost return (J_c)
        self.nu = self.nu - self.config['lr_nu'] * (self.config['cost_threshold'] - J_c_hat)
        self.nu = np.clip(self.nu, 0, self.config['nu_max'])

        dataset_size = obs.shape[0]
        indices = np.arange(dataset_size)
        
        # PPO-style Minibatch Updates
        for epoch in range(self.config['optim_epochs']):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.config['minibatch_size']): # steps of 64 to cover the whole batch of 2048
                end = start + self.config['minibatch_size']
                mb_idx = indices[start:end] # Select mini-batch from the full batch of data

                # Update Reward Critic
                v_r_pred = self.v_reward(obs[mb_idx]) # Critic estimates for rewards in the batch
                loss_v_r = nn.MSELoss()(v_r_pred, ret_r[mb_idx]) # Fit Critic's V_r(s) to the computed V_r targets (Discounted Rewards)
                self.v_r_optimizer.zero_grad() # Clear gradients
                loss_v_r.backward() # Backpropagate reward critic loss
                self.v_r_optimizer.step() # Update reward critic parameters

                # Update Cost Critic
                v_c_pred = self.v_cost(obs[mb_idx]) # Critic estimates for costs in the batch
                loss_v_c = nn.MSELoss()(v_c_pred, ret_c[mb_idx]) # Fit Critic's V_c(s) to the computed V_c targets (Discounted Costs)
                self.v_c_optimizer.zero_grad() # Clear gradients
                loss_v_c.backward() # Backpropagate cost critic loss
                self.v_c_optimizer.step() # Update cost critic parameters

                curr_dist = self.pi(obs[mb_idx]) # Gaussian distribution for current policy's mini-batch

                # Update Policy
                curr_logp = curr_dist.log_prob(acts[mb_idx]).sum(axis=-1) # Sum over action dimensions to get total log-prob
                ratio = torch.exp(curr_logp - old_logp[mb_idx])
                
                # Per-state KL Divergence
                mb_old_dist = Normal(old_dist.loc[mb_idx], old_dist.scale[mb_idx]) # Construct old distribution for mini-batch (Using mean and stddev)
                kl = torch.distributions.kl_divergence(mb_old_dist, curr_dist).sum(axis=-1)
                
                # Trust region mask
                valid_kl_mask = (kl <= self.config['trust_region_delta']).float()

                # FOCOPS objective
                combined_adv = adv_r[mb_idx] - self.nu * adv_c[mb_idx]
                loss_pi = (kl - (1 / self.config['temperature_lam']) * ratio * combined_adv) * valid_kl_mask
                loss_pi = loss_pi.mean()

                self.pi_optimizer.zero_grad()
                loss_pi.backward()
                self.pi_optimizer.step()

            # Early Stopping Check
            with torch.no_grad():
                current_dist_all = self.pi(obs)
                avg_kl = torch.distributions.kl_divergence(old_dist, current_dist_all).sum(axis=-1).mean().item()
            
            if avg_kl > self.config['trust_region_delta']:
                break # Trust region violated, stop epochs early