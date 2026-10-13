# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .mlp import MLP

class ActorNetwork(MLP):
    """Actor network using MLP."""

    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 hidden_dims: tuple = (256, 256, 256), 
                 activation_fn: type[nn.Module] = nn.ReLU,
                 output_activation_fn: type[nn.Module] = nn.Tanh,
                 name: str = "actor"):
        
        super().__init__(
            input_dim=obs_dim,
            output_dim=act_dim,
            hidden_dims=hidden_dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            name=name
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the actor network."""
        return super().forward(obs)
    

class CriticNetwork(MLP):
    """Critic network using MLP."""

    def __init__(self, 
                 obs_dim: int, 
                 hidden_dims: tuple = (256, 256, 256), 
                 activation_fn: type[nn.Module] = nn.ReLU,
                 output_activation_fn: type[nn.Module] = nn.Identity,
                 name: str = "critic"):
        
        super().__init__(
            input_dim=obs_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            name=name
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic network."""
        return super().forward(obs)


class ActorCritic(nn.Module):
    """Actor-Critic network using MLP for both actor and critic."""

    def __init__(self, 
                 obs_dim: int, 
                 act_dim: int, 
                 actor_hidden_dims: tuple = (256, 256, 256), 
                 actor_activation_fn: type[nn.Module] = nn.ReLU,
                 actor_output_activation_fn: type[nn.Module] = nn.Tanh,
                 critic_hidden_dims: tuple = (256, 256, 256), 
                 critic_activation_fn: type[nn.Module] = nn.ReLU,
                 critic_output_activation_fn: type[nn.Module] = nn.Identity):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim, actor_hidden_dims, actor_activation_fn, actor_output_activation_fn)
        self.critic = MLP(act_dim + obs_dim, 1, critic_hidden_dims, critic_activation_fn, critic_output_activation_fn)


    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from actor network."""
        with torch.no_grad():
            return self.actor(obs)


if __name__ == "__main__":
    obs_dim = 10
    act_dim = 2
    batch_size = 4

    ac = ActorCritic(obs_dim, act_dim)

    sample_obs = torch.randn(batch_size, obs_dim)
    actions = ac.act(sample_obs)
    q_values = ac.critic(torch.cat([sample_obs, actions], dim=-1))

    print("Sample Observations:", sample_obs)
    print("Sample Actions:", actions)
    print("Sample Q-values:", q_values)