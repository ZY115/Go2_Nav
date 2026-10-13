#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import torch
import warnings
import numpy as np
import torch.nn as nn
from typing import Union

from edge_drl_student.agents.off_policy_agent import OffPolicyAgent
from edge_drl_student.memory.buffers.base_buffer import TransitionBatch
from edge_drl_student.common.noise import Noise, OUNoise, GaussianNoise
from edge_drl_student.networks.actor_critic import ActorCritic, ActorNetwork, CriticNetwork
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.common.normalizer import BaseNormalizer, MinMaxNormalizer, EmpiricalNormalizer


class DDPGAgent(OffPolicyAgent):
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
                 use_adaptive_lr: bool = False,
                 actor_decay_gamma: float = 0.99995,
                 critic_decay_gamma: float = 0.9999,
                 actor_lr=1e-6,
                 critic_lr=1e-5,
                 gamma=0.95,
                 tau=0.005,
                 noise: Noise = None,
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
        
        # Learning rate schedulers
        if use_adaptive_lr:
            self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.actor_optimizer, gamma=actor_decay_gamma
            )
            self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.critic_optimizer, gamma=critic_decay_gamma
            )
        self.use_adaptive_lr = use_adaptive_lr
        
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
            
        # Noise process
        self.noise = noise if noise is not None else OUNoise(size=act_dim, device=device)
            
        # The paths for saving/loading models
        self.load_path = model_load_path
        self.save_path = model_save_path

        # Action bounds
        self.action_bound = action_bound
        
        self.to(device)
        
        # Print model summary
        self.print_model_summary()

    def train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()
        
    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def to(self, device: str = None):
        """Transfer networks to specified device."""
        device = torch.device(device) if device is not None else self.device
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)
        self.device = device
        
    def act(self, obs: torch.Tensor, explore: bool = True) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action: torch.Tensor = self.actor(obs)
            
        # Add exploration noise
        if explore:
            action += self.noise(action.shape)
            
        # Clip action to [-1, 1] for valid range
        action_clipped = torch.clamp(action, -1.0, 1.0)  
        
        # Return denormalized action
        return self.actor_normalizer.denormalize(action_clipped)

    def update(self, batch_size) -> tuple[torch.Tensor, torch.Tensor]:
        """DDPG update step."""
        
        t_batch = self.replay_buffer.sample(batch_size)
        obs, act, rew, next_obs, done = t_batch.obs, t_batch.action, t_batch.reward, t_batch.next_obs, t_batch.done

        with torch.no_grad():
            target_act = self.target_actor(next_obs)
            target_q = self.target_critic(self._critic_input(next_obs, target_act))
            target_val: torch.Tensor = rew + self.gamma * (1 - done) * target_q

        q: torch.Tensor = self.critic(self._critic_input(obs, act))
        critic_loss = ((q - target_val) ** 2).mean()

        # actor_loss = -self.critic(self._critic_input(obs, self.actor(obs))).mean()

        # Critic network update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor network update
        actor_actions = self.actor(obs)
        q_values: torch.Tensor = self.critic(self._critic_input(obs, actor_actions))
        actor_loss = -q_values.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update learning rate schedulers
        if self.use_adaptive_lr:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            
        self.noise.step()   # update noise process if needed

        self._soft_update(self.target_critic, self.critic, self.tau)
        self._soft_update(self.target_actor, self.actor, self.tau)
        
        print(f"[DDPG Update] Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        return actor_loss, critic_loss
        
        
    def save_models(self, path: str=None):
        """
        Save model parameters and optimizer states.
        Args:
            path (str): Directory path to save model files.
        """
        path = path if path is not None else self.save_path
        if path is None:
            warnings.warn("No save path specified for saving models.")
            return
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

    def load_models(self, path: str=None, strict: bool=True):
        """
        Load model and optimizer parameters.
        Args:
            path (str): Directory where models were saved.
            strict (bool): Whether to enforce strict loading.
        """
        path = path if path is not None else self.load_path
        if path is None:
            warnings.warn("⚠️ No load path specified for loading models.")
            return
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
        print(f"[DDPG] initialized on {self.device} (γ={self.gamma}, τ={self.tau})")
        print("\n ================ Actor Network ================ \n")
        print(self.actor)
        print(f"Total params: {sum(p.numel() for p in self.actor.parameters() if p.requires_grad):,}")

        print("\n ================ Critic Network ================ \n")
        print(self.critic)
        print(f"Total params: {sum(p.numel() for p in self.critic.parameters() if p.requires_grad):,}")


    def _critic_input(self, obs, act) -> torch.Tensor:
        """Prepare critic input by concatenating observation and action."""
        return torch.cat((obs, act), dim=-1)
        
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    
    obs_dim = 12
    act_dim = 6
    device = "cuda:0"
    action_bound = (
        torch.tensor([-10., -10., -10., -20., -20., -20.], device=device),
        torch.tensor([ 10.,  10.,  10.,  20.,  20.,  20.], device=device)
    )
    
    replay_buffer = SoloReplayBuffer(buffer_size=10000, 
                                     prefill_size=1000,
                                     obs_dim=obs_dim,
                                     action_dim=act_dim, 
                                     device=device)
    
    ddpg_agent = DDPGAgent(obs_dim=obs_dim,
                           act_dim=act_dim,
                           replay_buffer=replay_buffer,
                           action_bound=action_bound,
                           device=device)
    for _ in range(1500):
        transition = TransitionBatch(
            obs=torch.randn(1, obs_dim, device=device),
            action=torch.randn(1, act_dim, device=device),
            reward=torch.randn(1, 1, device=device),
            next_obs=torch.randn(1, obs_dim, device=device),
            done=torch.zeros(1, 1, device=device)
        )
        replay_buffer.add_transition(transition)    

    sample_obs = torch.randn(4, obs_dim, device=device)
    actions = ddpg_agent.act(sample_obs)
    
    actor_loss, critic_loss = ddpg_agent.update(batch_size=64)
    
    print("Sample Actions from DDPG Agent:", actions)
    print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")