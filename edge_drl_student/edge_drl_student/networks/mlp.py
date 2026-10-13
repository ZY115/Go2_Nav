# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
import torch.nn as nn
from typing import Type, Union
from abc import ABC, abstractmethod
from .base_network import BaseNetwork

class MLP(BaseNetwork):
    """Standard Multi-Layer Perceptron (MLP) network."""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: tuple[int] | list[int] = (256, 256, 256),
                 activation_fn: Union[Type[nn.Module], str, list[str]] = nn.ReLU,
                 output_activation_fn: Union[Type[nn.Module], str, None] = None,
                 name: str = "mlp"):
        """ Initialize MLP

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            hidden_dims (tuple[int] | list[int], optional): _description_. Defaults to (256, 256, 256).
            activation_fn (type[nn.Module], optional): _description_. Defaults to nn.ReLU.
            output_activation (type[nn.Module] | None, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "mlp".
        """
        self.hidden_dims = hidden_dims
        
        self.activation_fn = self._process_activation_fn(activation_fn, len(hidden_dims))
        self.output_activation_fn = self._process_output_activation(output_activation_fn)
        
        super().__init__(input_dim, output_dim, hidden_dims[0], name)
        

    def _init_layers(self):
        """Initialize MLP layers dynamically based on hidden_dims."""
        layers = []
        prev_dim = self.input_dim

        # Hidden layers
        for i, h_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self.activation_fn[i]())
            prev_dim = h_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(self.output_activation_fn())

        # if self.output_activation is not None:
        #     layers.append(self.output_activation())

        self.net = nn.Sequential(*layers)
        self.apply(self.init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.net(x)
    

    @classmethod
    def _process_activation_fn(cls, 
                               act_fn: Union[Type[nn.Module], str, list[Union[str, Type[nn.Module]]]],
                               num_layers: int
    ) -> list[Type[nn.Module]]:
        """Process activation function(s) for hidden layers. Support multiple input types.

        Args:
            act_fn: activation function(s). Can be:
                - a class (e.g. nn.ReLU)
                - a string (e.g. "relu")
                - a list of classes or strings
        Returns:
            A list of classes of nn.Module for each hidden layer.
        """
        # Case 1: string input
        if isinstance(act_fn, str):
            return [cls._activation_fn(act_fn) for _ in range(num_layers)]
        
        # Case 2: Single type (nn.Module subclass)
        elif isinstance(act_fn, type) and issubclass(act_fn, nn.Module):
            return [act_fn for _ in range(num_layers)]

        # Case 3: List input
        elif isinstance(act_fn, (list, tuple)):
            activation_fns = []
            for fn in act_fn:
                if isinstance(fn, str):
                    activation_fns.append(cls._activation_fn(fn))
                elif isinstance(fn, type) and issubclass(fn, nn.Module):
                    activation_fns.append(fn)
                else:
                    raise TypeError(f"Invalid activation type in list: {type(fn)}")
            return activation_fns

        else:
            raise TypeError(f"Unsupported activation type: {type(act_fn)}")


    @classmethod
    def _process_output_activation(cls, 
                                   output_act: Union[Type[nn.Module], str, None]
                                   ) -> Type[nn.Module]:
        """
        Process output activation function.
        
        Args:
            output_act: Output activation (class, string, or None)
            
        Returns:
            class of nn.Module
        """
        if output_act is None:
            return nn.Identity
        elif isinstance(output_act, str):
            return cls._activation_fn(output_act)
        elif isinstance(output_act, type) and issubclass(output_act, nn.Module):
            return output_act
        else:
            raise TypeError(
                f"Invalid output_activation type: {type(output_act)}. "
                f"Expected nn.Module class, string, or None."
            )

if __name__ == "__main__":
    # Simple test
    mlp = MLP(
        input_dim=10,
        output_dim=2,
        hidden_dims=[64, 128, 64],
        activation_fn=["elu", nn.ReLU, "tanh"],
        output_activation_fn='tanh'
    )
    x = torch.randn(4, 10)
    out = mlp(x)
    print(mlp.activation_fn)
    print(mlp.output_activation_fn)
    print(out.shape)  # Should be (4, 2)