# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Type

class BaseNetwork(nn.Module, ABC):
    """Abstract neural network structure used by all agents."""

    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 hidden_dim: int = 256, 
                 name: str = "neural network"):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.name = name
        self.iteration = 0
        self._init_layers()

    @abstractmethod
    def _init_layers(self):
        """Define layers in subclass."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass logic."""
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                nn.init.xavier_uniform_(m.weight)
                m.bias.fill_(0.01)   
            
    @classmethod
    def _activation_fn(cls, act_fn: str) -> Type[nn.Module]:
        act_fn = act_fn.lower()
        mapping = {
            "elu": nn.ELU,
            "selu": nn.SELU,
            "relu": nn.ReLU,
            "crelu": nn.ReLU,
            "lrelu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "linear": nn.Identity,
            "softmax": lambda: nn.Softmax(dim=-1),
        }
        if act_fn not in mapping:
            raise ValueError(f"❌ Unknown activation function: {act_fn}")
        return mapping[act_fn]