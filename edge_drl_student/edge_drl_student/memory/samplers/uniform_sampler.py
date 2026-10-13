# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
from .base_sampler import BaseSampler

class UniformSampler(BaseSampler):
    """
    UniformSampler
    ---------------
    A simple sampler that draws indices uniformly at random
    from the valid portion of the replay buffer.
    """

    def __init__(self, buffer_size: int, random_seed: int = 42, device: str = "cpu"):
        super().__init__(buffer_size)
        self.random_seed = random_seed
        self.device = device
        
        # independent RNG (does NOT affect global numpy random)
        self.rng = torch.Generator(device=self.device)

    def set_seed(self, seed: int):
        """Reset the sampler's random seed."""
        self.random_seed = seed
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)

    def fetch(self, batch_size: int, valid_size: int) -> torch.Tensor:
        """
        Uniformly sample indices from [0, valid_size).

        Args:
            batch_size (int): Number of samples to fetch.
            valid_size (int): Number of valid transitions currently stored.

        Returns:
            torch.Tensor: Randomly sampled indices.
        """
        if valid_size <= 0:
            raise ValueError("Cannot fetch from an empty buffer.")
        if valid_size > self.buffer_size:
            raise ValueError(
                f"valid_size ({valid_size}) cannot exceed buffer_size ({self.buffer_size})."
            )
        # Sample with replacement
        indices = torch.randint(0, valid_size, (batch_size,), generator=self.rng, device=self.device)
        return indices

    def update(self, indices: torch.Tensor, **kwargs):
        """Uniform sampler has no internal state to update."""
        pass
