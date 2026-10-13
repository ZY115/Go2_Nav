# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import abc
import torch
import torch.nn as nn
from typing import Dict
from collections import deque
from typing import Union, Tuple, Optional


# ============================================================
# ====                  Abstract Base Class               ====
# ============================================================
class BaseNormalizer(abc.ABC):
    """Abstract base class for all normalizers."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abc.abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        pass

    @abc.abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize input tensor."""
        pass

    def update(self, x: torch.Tensor):
        """Optional: Update normalizer statistics with new data."""
        pass

    def to(self, device: str):
        """Move normalizer to specified device."""
        self.device = device
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def state_dict(self):
        """Return state dictionary for saving."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def load_state_dict(self, state_dict: Dict):
        """Load state from dictionary."""
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
            setattr(self, k, v)

    def save(self, path: str):
        """Save normalizer state to file."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load normalizer state from file."""
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)


# ============================================================
# ======               Min–Max Normalizer               ======
# ============================================================
class MinMaxNormalizer(BaseNormalizer):
    """
    Linearly scales inputs from [min, max] → [-1, 1].
    
    Args:
        min_val: Minimum value(s) of the input range
        max_val: Maximum value(s) of the input range
        device: PyTorch device
    
    Example:
        >>> normalizer = MinMaxNormalizer(min_val=0.0, max_val=10.0)
        >>> x = torch.tensor([0.0, 5.0, 10.0])
        >>> normalized = normalizer.normalize(x)  # [-1.0, 0.0, 1.0]
    """

    def __init__(
        self, 
        min_val: Union[float, torch.Tensor], 
        max_val: Union[float, torch.Tensor], 
        device: str = "cpu"
    ):
        super().__init__(device)
        self.min = torch.as_tensor(min_val, dtype=torch.float32, device=device)
        self.max = torch.as_tensor(max_val, dtype=torch.float32, device=device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Scale from [min, max] to [-1, 1]."""
        with torch.no_grad():
            return 2.0 * (x - self.min) / (self.max - self.min + 1e-8) - 1.0

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Scale from [-1, 1] to [min, max]."""
        with torch.no_grad():
            return (x + 1.0) / 2.0 * (self.max - self.min) + self.min


# =============================================================
# ===   Running Mean/Std Normalizer (Welford's Algorithm) ====
# =============================================================
class RunningMeanStdNormalizer(BaseNormalizer):
    """
    Online z-score normalization using Welford's algorithm.
    
    Maintains running mean and variance without storing all historical data.
    All historical samples have equal weight in the statistics.
    
    Args:
        shape: Shape of the data to normalize (e.g., (obs_dim,) or ())
        device: PyTorch device
    
    Example:
        >>> normalizer = RunningMeanStdNormalizer(shape=(10,))
        >>> for batch in data_loader:
        ...     normalizer.update(batch)
        ...     normalized = normalizer.normalize(batch)
    
    Reference:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]], device: str = "cpu"):
        super().__init__(device)
        
        if isinstance(shape, int):
            shape = (shape,)
        
        self.shape = shape
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        """
        Update running statistics with new data using Welford's algorithm.
        
        Args:
            x: Input tensor of shape [batch_size, *shape] or [*shape]
        """
        with torch.no_grad():
            x = x.detach()
            
            # Handle single sample case
            if x.dim() == len(self.shape):
                x = x.unsqueeze(0)
            
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = float(x.shape[0])

            # Welford's online algorithm
            delta = batch_mean - self.mean
            total_count = self.count + batch_count

            # Update mean
            new_mean = self.mean + delta * (batch_count / total_count)

            # Update variance
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
            new_var = M2 / total_count

            # Inplace update
            self.mean.copy_(new_mean)
            self.var.copy_(new_var)
            self.count.add_(batch_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score normalization: (x - mean) / std."""
        with torch.no_grad():
            return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of normalization: x * std + mean."""
        with torch.no_grad():
            return x * torch.sqrt(self.var + 1e-8) + self.mean

    def reset(self):
        """Reset statistics to initial state."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.fill_(1e-4)

    def get_stats(self):
        """Get current statistics."""
        return {
            'mean': self.mean.clone(),
            'std': torch.sqrt(self.var + 1e-8),
            'var': self.var.clone(),
            'count': self.count.item(),
        }

    def __repr__(self):
        return (
            f"RunningMeanStdNormalizer(\n"
            f"  shape={self.shape},\n"
            f"  count={self.count.item():.0f},\n"
            f"  device='{self.device}'\n"
            f")"
        )


# =============================================================
# =====  Exponential Moving Average Normalizer (EMA)     =====
# =============================================================
class EMANormalizer(BaseNormalizer):
    """
    Exponential Moving Average normalization.
    
    Recent samples have higher weight than older samples.
    Useful when data distribution changes over time.
    
    Args:
        shape: Shape of the data to normalize
        momentum: Weight for old statistics (0.99 = slow adaptation)
        device: PyTorch device
    
    Example:
        >>> normalizer = EMANormalizer(shape=(10,), momentum=0.99)
        >>> normalizer.update(batch)  # Recent data weighted more
    """

    def __init__(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        momentum: float = 0.99, 
        device: str = "cpu"
    ):
        super().__init__(device)
        
        if isinstance(shape, int):
            shape = (shape,)
        
        self.shape = shape
        self.momentum = momentum
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(0.0, dtype=torch.float32, device=device)

    def update(self, x: torch.Tensor):
        """Update statistics using exponential moving average."""
        with torch.no_grad():
            x = x.detach()
            
            # Handle single sample
            if x.dim() == len(self.shape):
                x = x.unsqueeze(0)
            
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            if self.count == 0:
                # First update: use batch statistics
                self.mean.copy_(batch_mean)
                self.var.copy_(batch_var)
            else:
                # EMA update
                self.mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                self.var.mul_(self.momentum).add_(batch_var, alpha=1 - self.momentum)

            self.count.add_(x.shape[0])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score normalization."""
        with torch.no_grad():
            return (x - self.mean) / torch.sqrt(self.var + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        with torch.no_grad():
            return x * torch.sqrt(self.var + 1e-8) + self.mean

    def reset(self):
        """Reset statistics."""
        self.mean.zero_()
        self.var.fill_(1.0)
        self.count.zero_()

    def get_stats(self):
        """Get current statistics."""
        return {
            'mean': self.mean.clone(),
            'std': torch.sqrt(self.var + 1e-8),
            'var': self.var.clone(),
            'count': self.count.item(),
            'momentum': self.momentum,
        }

    def __repr__(self):
        return (
            f"EMANormalizer(\n"
            f"  shape={self.shape},\n"
            f"  momentum={self.momentum},\n"
            f"  count={self.count.item():.0f},\n"
            f"  device='{self.device}'\n"
            f")"
        )


# =============================================================
# =====      Empirical Normalizer (nn.Module version)    =====
# =============================================================
class EmpiricalNormalizer(BaseNormalizer, nn.Module):
    """
    Online normalization as nn.Module (can be part of a network).
    
    Supports training/eval modes and optional update limits.
    Uses Welford's algorithm with rate-based updates.
    
    Args:
        shape: Shape of the data to normalize
        eps: Small constant for numerical stability
        until: Stop updating after this many samples (None = unlimited)
        device: PyTorch device
    
    Example:
        >>> normalizer = EmpiricalNormalizer(shape=(10,))
        >>> normalizer.train()  # Updates statistics
        >>> output = normalizer(input)
        >>> normalizer.eval()   # Freezes statistics
    """

    def __init__(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        eps: float = 1e-6, 
        until: Optional[int] = None, 
        device: str = "cpu"
    ):
        BaseNormalizer.__init__(self, device)
        nn.Module.__init__(self)

        if isinstance(shape, int):
            shape = (shape,)

        self.shape = shape
        self.eps = eps
        self.until = until

        # Register as buffers (saved in state_dict, moved with .to())
        self.register_buffer("_mean", torch.zeros(shape, device=device))
        self.register_buffer("_var", torch.ones(shape, device=device))
        self.register_buffer("_std", torch.ones(shape, device=device))
        self.register_buffer("_count", torch.tensor(0.0, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: update if training, then normalize."""
        if self.training:
            self.update(x)
        return self.normalize(x)

    def update(self, x: torch.Tensor):
        """Update statistics using rate-based Welford's algorithm."""
        with torch.no_grad():
            x = x.detach()

            # Check update limit
            if self.until is not None and self._count.item() >= self.until:
                return

            # Handle single sample
            if x.dim() == len(self.shape):
                x = x.unsqueeze(0)

            batch_count = float(x.shape[0])
            self._count.add_(batch_count)

            # Compute update rate
            rate = batch_count / max(self._count.item(), 1e-8)

            # Batch statistics
            mean_x = x.mean(dim=0)
            var_x = x.var(dim=0, unbiased=False)

            # Welford's update
            delta_mean = mean_x - self._mean
            new_mean = self._mean + rate * delta_mean
            new_var = self._var + rate * (var_x - self._var + delta_mean * (mean_x - new_mean))

            # Inplace update
            self._mean.copy_(new_mean)
            self._var.copy_(torch.clamp(new_var, min=0.0))  # Ensure non-negative
            self._std.copy_(torch.sqrt(self._var + self.eps))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Z-score normalization."""
        return (x - self._mean) / (self._std + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        return x * (self._std + self.eps) + self._mean

    def reset(self):
        """Reset statistics."""
        self._mean.zero_()
        self._var.fill_(1.0)
        self._std.fill_(1.0)
        self._count.zero_()

    def get_stats(self):
        """Get current statistics."""
        return {
            'mean': self._mean.clone(),
            'std': self._std.clone(),
            'var': self._var.clone(),
            'count': self._count.item(),
            'until': self.until,
        }

    def __repr__(self):
        return (
            f"EmpiricalNormalizer(\n"
            f"  shape={self.shape},\n"
            f"  count={self._count.item():.0f},\n"
            f"  until={self.until},\n"
            f"  training={self.training},\n"
            f"  device='{self.device}'\n"
            f")"
        )


# =============================================================
# =====   Sliding Window Normalizer (Fixed-size Window)  =====
# =============================================================
class SlidingWindowNormalizer(BaseNormalizer):
    """
    Normalization using sliding window statistics.
    
    Maintains a fixed-size window of recent observations.
    Useful for non-stationary environments where old data should be forgotten.
    
    Args:
        shape: Shape of the data to normalize
        window_size: Maximum number of samples to keep
        clip_range: Clip normalized values to [-clip_range, clip_range]
        device: PyTorch device
    
    Example:
        >>> # For scalar rewards
        >>> reward_norm = SlidingWindowNormalizer(
        ...     shape=(), 
        ...     window_size=1000, 
        ...     clip_range=5.0
        ... )
        >>> 
        >>> # For vector observations
        >>> obs_norm = SlidingWindowNormalizer(
        ...     shape=(24,), 
        ...     window_size=10000
        ... )
    """

    def __init__(
        self, 
        shape: Union[int, Tuple[int, ...]], 
        window_size: int = 1000, 
        clip_range: Optional[float] = None,
        device: str = "cpu"
    ):
        super().__init__(device)

        if isinstance(shape, int):
            shape = (shape,) if shape > 0 else ()
        
        self.shape = shape
        self.window_size = window_size
        self.clip_range = clip_range

        # Sliding window storage
        self.window = deque(maxlen=window_size)

        # Cached statistics
        self.mean = torch.zeros(self.shape, dtype=torch.float32, device=device)
        self.std = torch.ones(self.shape, dtype=torch.float32, device=device)
        self._stats_dirty = True

    def update(self, x: torch.Tensor):
        """
        Add new sample(s) to the sliding window.
        
        Args:
            x: Input tensor of shape [batch_size, *shape] or [*shape]
        """
        with torch.no_grad():
            x = x.detach().cpu()

            # Handle both single and batched inputs
            if x.dim() == len(self.shape):
                # Single sample
                self.window.append(x.clone())
            else:
                # Batch of samples
                for i in range(x.shape[0]):
                    self.window.append(x[i].clone())

            self._stats_dirty = True

    def _compute_stats(self):
        """Compute mean and std from current window."""
        if not self._stats_dirty or len(self.window) == 0:
            return

        with torch.no_grad():
            if len(self.window) == 1:
                self.mean = self.window[0].to(self.device)
                self.std = torch.ones_like(self.mean)
            else:
                # Stack all samples: [window_size, *shape]
                window_tensor = torch.stack(list(self.window), dim=0)
                self.mean = window_tensor.mean(dim=0).to(self.device)
                self.std = window_tensor.std(dim=0, unbiased=False).to(self.device)

                # Avoid division by zero
                self.std = torch.clamp(self.std, min=1e-8)

            self._stats_dirty = False

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input using current window statistics.
        
        Returns:
            Z-score normalized tensor (optionally clipped)
        """
        self._compute_stats()

        # Z-score normalization
        normalized = (x - self.mean) / (self.std + 1e-8)

        # Optional clipping
        if self.clip_range is not None:
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)

        return normalized

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize input using current window statistics.
        
        Note: Approximate if clipping was applied during normalization.
        """
        self._compute_stats()
        return x * (self.std + 1e-8) + self.mean

    def reset(self):
        """Clear the sliding window and reset statistics."""
        self.window.clear()
        self.mean.zero_()
        self.std.fill_(1.0)
        self._stats_dirty = True

    def __len__(self):
        """Return current window size."""
        return len(self.window)

    def is_ready(self, min_samples: int = 2) -> bool:
        """
        Check if normalizer has enough samples for reliable statistics.
        
        Args:
            min_samples: Minimum number of samples required
        """
        return len(self.window) >= min_samples

    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'shape': self.shape,
            'window_size': self.window_size,
            'clip_range': self.clip_range,
            'window': [x.cpu() for x in self.window],
            'mean': self.mean.cpu(),
            'std': self.std.cpu(),
            'device': self.device,
        }

    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.shape = state_dict['shape']
        self.window_size = state_dict['window_size']
        self.clip_range = state_dict['clip_range']

        # Restore window
        self.window = deque(
            [x.to(self.device) for x in state_dict['window']],
            maxlen=self.window_size
        )

        # Restore statistics
        self.mean = state_dict['mean'].to(self.device)
        self.std = state_dict['std'].to(self.device)
        self._stats_dirty = False

    def get_stats(self):
        """Get current normalization statistics."""
        self._compute_stats()
        return {
            'mean': self.mean.clone(),
            'std': self.std.clone(),
            'window_size': len(self.window),
            'max_window_size': self.window_size,
            'clip_range': self.clip_range,
        }

    def __repr__(self):
        return (
            f"SlidingWindowNormalizer(\n"
            f"  shape={self.shape},\n"
            f"  window_size={self.window_size},\n"
            f"  current_size={len(self.window)},\n"
            f"  clip_range={self.clip_range},\n"
            f"  device='{self.device}'\n"
            f")"
        )


# =============================================================
# =====              Identity Normalizer                 =====
# =============================================================
class IdentityNormalizer(BaseNormalizer):
    """
    No-op normalizer (returns input unchanged).
    
    Useful as a placeholder or for ablation studies.
    
    Example:
        >>> normalizer = IdentityNormalizer()
        >>> x = torch.randn(10, 5)
        >>> assert torch.allclose(normalizer.normalize(x), x)
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged."""
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Return input unchanged."""
        return x

    def __repr__(self):
        return "IdentityNormalizer()"


# =============================================================
# =====              Utility Functions                   =====
# =============================================================
def create_normalizer(
    normalizer_type: str,
    shape: Union[int, Tuple[int, ...]],
    device: str = "cpu",
    **kwargs
) -> BaseNormalizer:
    """
    Factory function to create normalizers.
    
    Args:
        normalizer_type: Type of normalizer ('running', 'ema', 'sliding', 'minmax', 'identity')
        shape: Shape of data to normalize
        device: PyTorch device
        **kwargs: Additional arguments for specific normalizers
    
    Returns:
        Normalizer instance
    
    Example:
        >>> norm = create_normalizer('running', shape=(10,), device='cuda')
        >>> norm = create_normalizer('sliding', shape=(), window_size=1000)
    """
    normalizer_type = normalizer_type.lower()
    
    if normalizer_type in ['running', 'runningmeanstd']:
        return RunningMeanStdNormalizer(shape=shape, device=device)
    
    elif normalizer_type == 'ema':
        momentum = kwargs.get('momentum', 0.99)
        return EMANormalizer(shape=shape, momentum=momentum, device=device)
    
    elif normalizer_type in ['sliding', 'window']:
        window_size = kwargs.get('window_size', 1000)
        clip_range = kwargs.get('clip_range', None)
        return SlidingWindowNormalizer(
            shape=shape, 
            window_size=window_size, 
            clip_range=clip_range, 
            device=device
        )
    
    elif normalizer_type == 'empirical':
        eps = kwargs.get('eps', 1e-6)
        until = kwargs.get('until', None)
        return EmpiricalNormalizer(shape=shape, eps=eps, until=until, device=device)
    
    elif normalizer_type == 'minmax':
        min_val = kwargs.get('min_val', -1.0)
        max_val = kwargs.get('max_val', 1.0)
        return MinMaxNormalizer(min_val=min_val, max_val=max_val, device=device)
    
    elif normalizer_type in ['identity', 'none']:
        return IdentityNormalizer(device=device)
    
    else:
        raise ValueError(
            f"Unknown normalizer type: {normalizer_type}. "
            f"Choose from: 'running', 'ema', 'sliding', 'empirical', 'minmax', 'identity'"
        )


# =============================================================
# =====                  Testing Code                     =====
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Normalizers")
    print("=" * 60)

    # Generate test data
    torch.manual_seed(42)
    data = torch.randn(1000, 5) * 2 + 3  # mean≈3, std≈2

    # Test 1: RunningMeanStdNormalizer
    print("\n1. RunningMeanStdNormalizer")
    print("-" * 60)
    norm1 = RunningMeanStdNormalizer(shape=(5,))
    
    for i in range(len(data)):
        norm1.update(data[i:i+1])
    
    stats = norm1.get_stats()
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")
    print(f"Count: {stats['count']}")
    
    normalized = norm1.normalize(data[:5])
    print(f"Normalized sample:\n{normalized}")

    # Test 2: EMANormalizer
    print("\n2. EMANormalizer")
    print("-" * 60)
    norm2 = EMANormalizer(shape=(5,), momentum=0.99)
    
    for i in range(len(data)):
        norm2.update(data[i:i+1])
    
    stats = norm2.get_stats()
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")

    # Test 3: SlidingWindowNormalizer
    print("\n3. SlidingWindowNormalizer")
    print("-" * 60)
    norm3 = SlidingWindowNormalizer(shape=(5,), window_size=100, clip_range=3.0)
    
    for i in range(len(data)):
        norm3.update(data[i:i+1])
    
    stats = norm3.get_stats()
    print(f"Window size: {stats['window_size']}/{stats['max_window_size']}")
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")

    # Test 4: Save and Load
    print("\n4. Save and Load")
    print("-" * 60)
    norm1.save("normalizer_test.pth")
    
    norm_loaded = RunningMeanStdNormalizer(shape=(5,))
    norm_loaded.load("normalizer_test.pth")
    
    print(f"Original mean: {norm1.mean}")
    print(f"Loaded mean:   {norm_loaded.mean}")
    print(f"Match: {torch.allclose(norm1.mean, norm_loaded.mean)}")

    # Test 5: Factory function
    print("\n5. Factory Function")
    print("-" * 60)
    norm_factory = create_normalizer('running', shape=(5,))
    print(norm_factory)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
