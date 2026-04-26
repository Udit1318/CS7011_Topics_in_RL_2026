import numpy as np
import torch
import scipy.signal

def discount_cumsum(x, discount):
    """
    Compute the discounted cumulative sum of a vector using a digital filter.

    Args:
        x (array-like): Input array of values to be discounted and summed.
        discount (float): Discount factor in the range [0, 1].

    Returns:
        ndarray: Discounted cumulative sum of the input array.

    Steps:
        1. Reverse the input array x
        2. Apply linear filter with numerator [1] and denominator [1, -discount].
            A linear filter looks like H(z) = B(z) / A(z), where B and A are polynomials. In this case, B(z) = 1 and A(z) = 1 - discount * z^-1.
            eg: For discount=0.99, the filter computes:
                y[t] = x[t] + 0.99 * y[t-1] 
        3. Reverse the filtered result back to original order
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95): # Discount Factor = 0.99, GAE Smoothing Lambda = 0.95
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_r_buf = np.zeros(size, dtype=np.float32)
        self.adv_c_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.cost_buf = np.zeros(size, dtype=np.float32)
        self.ret_r_buf = np.zeros(size, dtype=np.float32)
        self.ret_c_buf = np.zeros(size, dtype=np.float32)
        self.val_r_buf = np.zeros(size, dtype=np.float32)
        self.val_c_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, cost, val_r, val_c, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.val_r_buf[self.ptr] = val_r
        self.val_c_buf[self.ptr] = val_c
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val_r=0, last_val_c=0):
        # Define slice for the current path
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # Grab rewards and costs for the path
        rews = np.append(self.rew_buf[path_slice], last_val_r) # Append V_r(s') to r values
        vals_r = np.append(self.val_r_buf[path_slice], last_val_r) # Append V_r(s') to V_r values
        costs = np.append(self.cost_buf[path_slice], last_val_c) # Append V_c(s') to cost values
        vals_c = np.append(self.val_c_buf[path_slice], last_val_c) # Append V_c(s') to V_c values
        
        # Store Reward GAE and Returns for that path
        adv_r = rews[:-1] + self.gamma * vals_r[1:] - vals_r[:-1] # Q - V for rewards
        self.adv_r_buf[path_slice] = discount_cumsum(adv_r, self.gamma * self.lam) # Store GAE for rewards for the path
        self.ret_r_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # Store Sum of Discounted Rewards for the path
        
        # Calculate Cost GAE and Returns
        deltas_c = costs[:-1] + self.gamma * vals_c[1:] - vals_c[:-1] # Q - V for costs
        self.adv_c_buf[path_slice] = discount_cumsum(deltas_c, self.gamma * self.lam) # Store GAE for costs for the path
        self.ret_c_buf[path_slice] = discount_cumsum(costs, self.gamma)[:-1] # Store Sum of Discounted Costs for the path
        
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size # Buffer must be full
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize reward advantage
        adv_r_mean, adv_r_std = np.mean(self.adv_r_buf), np.std(self.adv_r_buf)
        self.adv_r_buf = (self.adv_r_buf - adv_r_mean) / (adv_r_std + 1e-8)

        # Normalize cost advantage
        adv_c_mean, adv_c_std = np.mean(self.adv_c_buf), np.std(self.adv_c_buf)
        self.adv_c_buf = (self.adv_c_buf - adv_c_mean) / (adv_c_std + 1e-8)
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret_r=self.ret_r_buf,
                    ret_c=self.ret_c_buf, adv_r=self.adv_r_buf, adv_c=self.adv_c_buf,
                    logp=self.logp_buf, raw_costs=self.cost_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}