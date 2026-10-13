# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

from __future__ import annotations

import torch
from abc import ABC, abstractmethod

class BaseSampler(ABC):
    """
    Abstract base class for all replay buffer samplers.

    A sampler defines how indices are drawn from the replay buffer.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        

    @abstractmethod
    def fetch(self, batch_size: int, valid_size: int) -> torch.Tensor:
        """
        Fetch a batch of sample indices from the buffer.

        Args:
            batch_size (int): Number of samples to fetch.
            valid_size (int): Number of valid transitions currently in the buffer.

        Returns:
            torch.Tensor: Sampled indices.
        """
        raise NotImplementedError
    

    @abstractmethod
    def update(self, indices: torch.Tensor, **kwargs):
        """
        Optionally update sampler state (e.g., priorities).
        For uniform samplers, this can be a no-op.
        """
        raise NotImplementedError


    def set_buffer_size(self, new_size: int):
        """Update buffer size dynamically (useful for circular buffers)."""
        self.buffer_size = new_size
        

    def __len__(self):
        return self.buffer_size
