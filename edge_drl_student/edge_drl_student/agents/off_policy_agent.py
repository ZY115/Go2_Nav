# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
import torch.nn as nn

from .base_agent import BaseAgent
from edge_drl_student.memory.buffers.base_buffer import TransitionBatch
from edge_drl_student.memory.buffers.off_policy.solo_replay_buffer import SoloReplayBuffer
from edge_drl_student.memory.buffers.off_policy.dual_replay_buffer import DualReplayBuffer
from edge_drl_student.memory.buffers.off_policy.cognitive_replay_buffer import CognitiveReplayBuffer
from typing import Union


class OffPolicyAgent(BaseAgent):
    """Base class for off-policy algorithms (DDPG, TD3, SAC, etc.)."""

    def __init__(self, 
                 replay_buffer: Union[SoloReplayBuffer, DualReplayBuffer, CognitiveReplayBuffer], 
                 batch_size=256,
                 gamma=0.95, 
                 tau=0.005,  
                 device="cpu"):
        super().__init__(device=device)
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

    # ==========================================================
    # Common behavior (shared by all off-policy agents)
    # ==========================================================
    def act(self, obs: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """Return action from the current policy."""
        raise NotImplementedError("Subclasses must implement act().")
    
    def sample_batch(self) -> TransitionBatch:
        return self.replay_buffer.sample(self.batch_size)

    def train_step(self, *args, **kwargs):
        """Abstract method — implemented in specific algorithm."""
        raise NotImplementedError("Subclasses must implement train_step().")
    
    @staticmethod
    @torch.no_grad()
    def _soft_update(target: nn.Module, source: nn.Module, tau: float):
        """Soft update target parameters."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.lerp_(sp.data, tau)

    @staticmethod
    @torch.no_grad()
    def _hard_update(target: nn.Module, source: nn.Module):
        """Hard copy parameters."""
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)


