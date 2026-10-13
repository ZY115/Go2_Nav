# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Minimal abstract base class for RL agents.
    Defines only the essential interfaces and utilities.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    # Abstract interfaces
    @abstractmethod
    def train_mode(self):
        """Set networks to training mode."""
        pass
        
    @abstractmethod
    def eval_mode(self):
        """Set networks to evaluation mode."""
        pass
    
    @abstractmethod
    def act(self, obs: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """Select action given observation."""
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs):
        """Update agent parameters."""
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        """Perform one gradient update."""
        pass
    
    @abstractmethod
    def save_models(self, path: str):
        """Abstract method — implemented in specific algorithm."""
        raise NotImplementedError("Subclasses must implement save_models().")
    
    @abstractmethod
    def load_models(self, path: str):
        """Abstract method — implemented in specific algorithm."""
        raise NotImplementedError("Subclasses must implement load_models().")

