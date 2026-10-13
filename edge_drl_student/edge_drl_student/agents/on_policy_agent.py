# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
import torch.nn as nn
from typing import Optional
from __future__ import annotations
from abc import ABC, abstractmethod

from base_agent import BaseAgent
from edge_drl_student.memory.buffers.on_policy.rollout_buffer import RolloutBuffer


class OnPolicyAgent(BaseAgent, ABC):
    """
    Base class for on-policy algorithms (PPO, A2C, TRPO, etc.).

    This class defines common logic such as:
        - rollout storage
        - advantage / return computation
        - general policy update interface
    """

    def __init__(
        self,
        rollout_buffer: RolloutBuffer,
        gamma: float = 0.99,
        lam: float = 0.95,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.rollout_buffer = rollout_buffer
        self.gamma = gamma
        self.lam = lam  # for GAE
        self.batch_size = batch_size
        self.device = torch.device(device)

    # ==========================================================
    # 🔹 Basic interaction API
    # ==========================================================
    def act(self, obs, deterministic: bool = False):
        """Return action from the current policy."""
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist, value = self.actor_critic(obs)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action.cpu().numpy(), value.cpu().numpy(), log_prob.cpu().numpy()

    def store_transition(self, obs, act, rew, done, val, log_prob):
        """Store trajectory step."""
        self.rollout_buffer.add_transition(obs, act, rew, done, val, log_prob)

    def sample_batch(self):
        """Sample minibatch from rollout buffer."""
        return self.rollout_buffer.sample(self.batch_size)

    # ==========================================================
    # 🔹 Common utilities (shared by PPO, A2C, etc.)
    # ==========================================================
    def compute_returns_and_advantages(self, last_value: torch.Tensor):
        """
        Compute GAE-Lambda and discounted returns.
        To be called after collecting one full rollout.
        """
        self.rollout_buffer.compute_gae(last_value, self.gamma, self.lam)

    @staticmethod
    def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute explained variance (diagnostic metric)."""
        var_y = torch.var(y_true)
        return torch.tensor(1.0) - torch.var(y_true - y_pred) / (var_y + 1e-8)

    # ==========================================================
    # 🔹 Abstract update method — must be implemented
    # ==========================================================
    @abstractmethod
    def train_step(self):
        """Perform one policy/value update using data from rollout buffer."""
        pass
