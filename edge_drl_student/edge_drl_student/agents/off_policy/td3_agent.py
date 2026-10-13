#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import torch
import warnings
import torch.nn as nn
from typing import Union

from edge_drl_student.agents.off_policy_agent import OffPolicyAgent
from edge_drl_student.memory.buffers.base_buffer import TransitionBatch
from edge_drl_student.networks.actor_critic import ActorCritic, ActorNetwork, CriticNetwork
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.common.normalizer import BaseNormalizer, MinMaxNormalizer, EmpiricalNormalizer


class TD3Agent(OffPolicyAgent):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 replay_buffer: Union[SoloReplayBuffer, DualReplayBuffer],
                 action_bound: tuple,
                 actor_hidden_dims: tuple = (256, 256, 256),
                 actor_activation_fn: type[nn.Module] = nn.ReLU,
                 actor_normalizer: BaseNormalizer = None,
                 critic_hidden_dims: tuple = (256, 256, 256),
                 critic_activation_fn: type[nn.Module] = nn.ReLU,
                 critic_normalizer: BaseNormalizer = None,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 model_load_path: str = None,
                 model_save_path: str = None,
                 device="cpu"):
        
        super().__init__(replay_buffer=replay_buffer, gamma=gamma, tau=tau, device=device)

        # === Networks ===
        self.actor = ActorNetwork(obs_dim, act_dim, actor_hidden_dims, actor_activation_fn, nn.Tanh)
        self.critic = CriticNetwork(obs_dim + act_dim, critic_hidden_dims, critic_activation_fn, nn.Identity)
        self.target_actor = ActorNetwork(obs_dim, act_dim, actor_hidden_dims, actor_activation_fn, nn.Tanh)
        self.target_critic = CriticNetwork(obs_dim + act_dim, critic_hidden_dims, critic_activation_fn, nn.Identity)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Normalizers
        if actor_normalizer is None:
            self.actor_normalizer = MinMaxNormalizer(min_val=action_bound[0],
                                                     max_val=action_bound[1], 
                                                     device=device)
        else:
            self.actor_normalizer = actor_normalizer

        if critic_normalizer is None:
            self.critic_normalizer = EmpiricalNormalizer(shape= (obs_dim + act_dim,), device=device)
        else:
            self.critic_normalizer = critic_normalizer
            
        # The paths for saving/loading models
        self.load_path = model_load_path
        self.save_path = model_save_path

        # Action bounds
        self.action_bound = action_bound
        
        self.to(device)
        
        
        self.print_model_summary()

    def to(self, device: str = None):
        device = torch.device(device) if device is not None else self.device
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        
    def act(self, obs: torch.Tensor, noise_scale=0.1, explore: bool = True) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(obs)
        if explore:
            action += noise_scale * torch.randn_like(action)
        action_clipped = torch.clamp(action, -1.0, 1.0)
        return self.actor_normalizer.denormalize(action_clipped)

    def update(self, batch_size, policy_delay=2):
        """TD3 update step."""
        batch = self.replay_buffer.sample(batch_size)
        s, a, r, s2, done = batch.obs, batch.action, batch.reward, batch.next_obs, batch.done

        # ==== 1. Target policy with smoothing ==== 
        noise = (torch.randn_like(a) * 0.2).clamp(-0.5, 0.5)
        a2 = (self.target_actor(s2) + noise).clamp(-1, 1)

        # ==== 2. Compute target Q using double critics ====
        target_Q1 = self.target_critic1(s2, a2)
        target_Q2 = self.target_critic2(s2, a2)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = r + self.gamma * (1 - done) * target_Q

        # ==== 3. Critic update ====
        Q1 = self.critic1(s, a)
        Q2 = self.critic2(s, a)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ==== 4. Delayed actor update ====
        if self.total_it % policy_delay == 0:
            actor_loss = -self.critic1(s, self.actor(s)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ==== 5. Soft update ====
            self._soft_update(self.target_critic1, self.critic1, self.tau)
            self._soft_update(self.target_critic2, self.critic2, self.tau)

        
        
    def save_models(self, path: str):
        """
        Save model parameters and optimizer states.
        Args:
            path (str): Directory path to save model files.
        """
        os.makedirs(path, exist_ok=True)
        try:
            torch.save(self.actor.state_dict(), os.path.join(path, "actor.pt"))
            torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
            torch.save(self.target_actor.state_dict(), os.path.join(path, "target_actor.pt"))
            torch.save(self.target_critic.state_dict(), os.path.join(path, "target_critic.pt"))

            torch.save(self.actor_optimizer.state_dict(), os.path.join(path, "actor_optim.pt"))
            torch.save(self.critic_optimizer.state_dict(), os.path.join(path, "critic_optim.pt"))

            # optional: save normalizers
            if hasattr(self, "actor_normalizer"):
                torch.save(self.actor_normalizer.state_dict(), os.path.join(path, "actor_norm.pt"))
            if hasattr(self, "critic_normalizer"):
                torch.save(self.critic_normalizer.state_dict(), os.path.join(path, "critic_norm.pt"))

            print(f"[✅] Models saved successfully to '{path}'")
        except Exception as e:
            warnings.warn(f"Failed to save models: {e}")

    def load_models(self, path: str, strict: bool = True):
        """
        Load model and optimizer parameters.
        Args:
            path (str): Directory where models were saved.
            strict (bool): Whether to enforce strict loading.
        """
        try:
            self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"), map_location=self.device), strict=strict)
            self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt"), map_location=self.device), strict=strict)
            self.target_actor.load_state_dict(torch.load(os.path.join(path, "target_actor.pt"), map_location=self.device), strict=strict)
            self.target_critic.load_state_dict(torch.load(os.path.join(path, "target_critic.pt"), map_location=self.device), strict=strict)

            # load optimizer states if exist
            actor_optim_path = os.path.join(path, "actor_optim.pt")
            critic_optim_path = os.path.join(path, "critic_optim.pt")
            if os.path.exists(actor_optim_path):
                self.actor_optimizer.load_state_dict(torch.load(actor_optim_path, map_location=self.device))
            if os.path.exists(critic_optim_path):
                self.critic_optimizer.load_state_dict(torch.load(critic_optim_path, map_location=self.device))

            # optional: load normalizers
            actor_norm_path = os.path.join(path, "actor_norm.pt")
            critic_norm_path = os.path.join(path, "critic_norm.pt")
            if hasattr(self, "actor_normalizer") and os.path.exists(actor_norm_path):
                self.actor_normalizer.load_state_dict(torch.load(actor_norm_path))
            if hasattr(self, "critic_normalizer") and os.path.exists(critic_norm_path):
                self.critic_normalizer.load_state_dict(torch.load(critic_norm_path))

            print(f"[✅] Models loaded successfully from '{path}'")
        except Exception as e:
            warnings.warn(f"⚠️ Failed to load models: {e}")
            
    def print_model_summary(self):
        print(f"[TD3] initialized on {self.device} (γ={self.gamma}, τ={self.tau})")
        
        print("\n ================ Actor Network ================ \n")
        print(self.actor)
        print(f"Total params: {sum(p.numel() for p in self.actor.parameters() if p.requires_grad):,}")

        print("\n ================ Critic Network ================ \n")
        print(self.critic)
        print(f"Total params: {sum(p.numel() for p in self.critic.parameters() if p.requires_grad):,}")

        
if __name__ == "__main__":
    obs_dim = 12
    act_dim = 6
    device = "cuda:0"
    action_bound = (
        torch.tensor([-10., -10., -10., -20., -20., -20.], device=device),
        torch.tensor([ 10.,  10.,  10.,  20.,  20.,  20.], device=device)
    )
    
    replay_buffer = SoloReplayBuffer(buffer_size=10000, obs_dim=obs_dim, action_dim=act_dim, device=device)
    td3_agent = TD3Agent(obs_dim=obs_dim,
                           act_dim=act_dim,
                           replay_buffer=replay_buffer,
                           action_bound=action_bound,
                           device=device)
    

    sample_obs = torch.randn(4, obs_dim, device=device)
    actions = td3_agent.act(sample_obs)
    print("Sample Actions from TD3 Agent:", actions)