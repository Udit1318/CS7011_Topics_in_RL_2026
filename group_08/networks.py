import torch
import torch.nn as nn
from torch.distributions import Normal

def mlp(sizes, activation, output_activation=nn.Identity):
    """Helper function to build a multi-layer perceptron."""
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# Architecture: 2 hidden layers with 64 units each between which we apply Tanh
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], nn.Tanh)
        # Initial log std set to -0.5 as per Table 3
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs):
        raw_mu = self.net(obs)
        # Strictly bound the mean between [-1, 1]
        mu = torch.tanh(raw_mu)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [1], nn.Tanh)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)