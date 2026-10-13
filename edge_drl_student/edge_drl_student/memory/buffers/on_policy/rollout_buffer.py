# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
from typing import Optional
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.buffers import TransitionBatch


class RolloutBuffer(BaseBuffer):
    """
    RolloutBuffer
    -------------
    A buffer for on-policy methods like PPO, A2C.
    It stores full rollouts (trajectories) collected from the latest policy.
    """

    def __init__(self, 
                 buffer_size: int,
                 obs_dim: int,
                 action_dim: int,
                 device: str = "cpu"):
        super().__init__(buffer_size, sampler=None, device=device)

        # Core experience fields
        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        # Computed after rollout ends
        self.advantages = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.returns = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    # ============================================================
    # Core API
    # ============================================================

    def add_transition(self, obs, act, rew, done, value, logp):
        """
        Add one transition (step) to the buffer.
        Called every environment step during rollout collection.
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = logp

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        """
        Compute GAE advantage and discounted returns.
        Typically called after finishing one rollout (horizon steps).
        """
        adv = 0
        for t in reversed(range(self.buffer_size)):
            # If it's the end of an episode, reset the advantage
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * last_value * mask - self.values[t]
            adv = delta + gamma * lam * mask * adv
            self.advantages[t] = adv
            self.returns[t] = adv + self.values[t]
            last_value = self.values[t]

        # Normalize advantages (common trick)
        adv_mean, adv_std = self.advantages.mean(), self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    def get(self, batch_size=None, shuffle=True):
        """
        Yield minibatches for PPO update.
        If batch_size is None → return full batch.
        """
        total_size = self.buffer_size if self.full else self.ptr

        indices = torch.arange(total_size, device=self.device)
        if shuffle:
            indices = indices[torch.randperm(total_size)]

        if batch_size is None:
            yield self._get_batch(indices)
        else:
            for start in range(0, total_size, batch_size):
                end = start + batch_size
                yield self._get_batch(indices[start:end])

    def _get_batch(self, indices):
        """Convert selected indices into TransitionBatch."""
        return TransitionBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=None,  # On-policy buffer doesn't store next_obs explicitly
            dones=self.dones[indices],
            infos=None,
        )

    def clear(self):
        """Reset buffer state after one full rollout."""
        self.ptr = 0
        self.full = False
        self.path_start_idx = 0
