# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

from __future__ import annotations

import csv
import torch
import pickle
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass

from edge_drl_student.memory.samplers import BaseSampler
from edge_drl_student.memory.samplers import UniformSampler


@dataclass
class Transition:
    """Batch-level semantic wrapper for sampled experiences."""
    obs: torch.Tensor       # Observation
    action: torch.Tensor    # Action taken
    reward: torch.Tensor    # Reward received
    next_obs: torch.Tensor  # Next observation
    done: torch.Tensor      # Whether the episode is done
    info: Any = None        # Additional info


@dataclass
class TransitionBatch:
    """Batched transitions sampled from the replay buffer."""
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor
    info: Optional[List[Any]] = None

    def to(self, device: str) -> "TransitionBatch":
        """Move all tensors to target device."""
        return TransitionBatch(
            obs=self.obs.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            next_obs=self.next_obs.to(device),
            done=self.done.to(device),
            info=self.info
        )

    def __len__(self):
        """Batch size."""
        return self.obs.shape[0]
    

class BaseBuffer(ABC):
    
    """
    BaseBuffer (abstract class)
    ---------------------------
    A lightweight and extendable replay buffer interface for single-environment use.

    Handles storage management (idx, full, buffer_size) and delegates
    sampling logic to a pluggable Sampler.
    """

    def __init__(self, 
                 buffer_size: int, 
                 prefill_size: int,
                 sampler: Optional[BaseSampler] = None, 
                 name: str = "BaseBuffer",
                 device: Optional[str] = 'cpu'):
        """
        Args:
            buffer_size (int): Maximum number of transitions the buffer can store.
            prefill_size (int): Number of transitions to prefill before sampling.
            sampler (Optional[BaseSampler]): A sampler used to fetch indices (default: UniformSampler).
            device (Optional[str]): The device to store the buffer on (default: 'cpu').
        """
        
        assert buffer_size > 0, "Buffer size must be positive."
        assert prefill_size >= 0, "Prefill size must be non-negative."
        assert prefill_size <= buffer_size, "Prefill size cannot exceed buffer size."    
        
        self._idx = 0
        self._full = False
        self._history_count = 0
        self.buffer_size = buffer_size
        self.prefill_size = prefill_size
        self.device = device
        self.buf_name = name
        
        self.sampler: BaseSampler = sampler or UniformSampler(buffer_size=buffer_size, 
                                                              random_seed=42, device=device)
        if hasattr(self.sampler, "set_buffer_size"):
            self.sampler.set_buffer_size(buffer_size)

    @abstractmethod
    def add_transition(self, *args, **kwargs):
        """
        Add one transition into the buffer.
        Typically called at every environment step.

        Subclasses may extend arguments, e.g.:</|endoftext|>
        - SoloReplayBuffer.add_transition(transition)
        - DualReplayBuffer.add_transition(transition, action_mode)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions using the attached sampler.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            A batch of transitions obtained via `_get_samples()`.
        """
        raise NotImplementedError

    def to_transition(self, obs, action, reward, next_obs, done, info=None) -> Transition:
        """Convert the tuples to Transition version."""

        def to_torch(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=self.device, dtype=torch.float32)
            return torch.as_tensor(x, device=self.device, dtype=torch.float32)

        return Transition(
            obs=to_torch(obs),
            action=to_torch(action),
            reward=to_torch(reward),
            next_obs=to_torch(next_obs),
            done=to_torch(done),
            info=to_torch(info) if info is not None else None,
        )
        
    def _get_samples(self, indices: torch.Tensor):
        """
        Given sampled indices, return the corresponding transitions.
        Should return a structured batch (e.g., dict, dataclass, namedtuple).
        """
        raise NotImplementedError
        
    def gather(self, indices: torch.Tensor) -> TransitionBatch:
        """Public API to gather by indices (reserved for external use)."""
        return self._get_samples(indices)    
        
    def extend(self, t_batch: TransitionBatch):
        """Add multiple transitions into the buffer."""
        raise NotImplementedError("extend() not implemented in BaseBuffer.")

    def set_sampler(self, sampler: BaseSampler):
        """Replace the sampler dynamically (e.g., switch to prioritized)."""
        self.sampler = sampler
        self.sampler.set_buffer_size(self.buffer_size)

    # ======================================================
    # Dump / Load Public API
    # ======================================================
    def dump(self, path: Union[str, Path], format: str = 'torch', compress: bool = False):
        """Save buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'torch':
            self._dump_torch(path, compress)
        elif format == 'pickle':
            self._dump_pickle(path)
        elif format == 'csv':
            self._dump_csv(path)
        else:
            raise ValueError(f"Unknown dump format {format}")

        print(f"Saved buffer data to {path}")

    def load(self, 
             path: Union[str, Path],
             format: str = 'auto',
             device: Optional[str] = None):
        """Load buffer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        # auto detect
        if format == 'auto':
            ext = path.suffix.lower()
            if ext in ['.pt', '.pth']:
                format = 'torch'
            elif ext in ['.pkl', '.pickle']:
                format = 'pickle'
            elif ext == '.csv':
                format = 'csv'
            else:
                raise ValueError(f"Could not auto-detect format from {ext}")

        # dispatch
        if format == 'torch':
            self._load_torch(path, device)
        elif format == 'pickle':
            self._load_pickle(path)
        elif format == 'csv':
            self._load_csv(path)
        else:
            raise ValueError(f"Unknown load format {format}")

        print(f"Loaded buffer ← {path}")

    # ======================================================
    # TORCH FORMAT
    # ======================================================
    def _dump_torch(self, path: Path, compress: bool):
        state = {
            # data
            'obs': self.obs,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_obs': self.next_obs,
            'dones': self.dones,

            # metadata
            'idx': self._idx,
            'full': self._full,
            'history_count': self._history_count,

            'buffer_size': self.buffer_size,
            'prefill_size': self.prefill_size,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,

            'sampler_state': getattr(self.sampler, "state_dict", lambda: {})()
        }

        torch.save(
            state,
            path,
            _use_new_zipfile_serialization=compress
        )

    def _load_torch(self, path: Path, device):
        data = torch.load(path, map_location=device or self.device)

        # restore tensors
        self.obs = data['obs'].to(self.device)
        self.actions = data['actions'].to(self.device)
        self.rewards = data['rewards'].to(self.device)
        self.next_obs = data['next_obs'].to(self.device)
        self.dones = data['dones'].to(self.device)

        # metadata
        self._idx = data['idx']
        self._full = data['full']
        self._history_count = data.get("history_count", self._history_count)

        # sanity restore dimensions
        self.obs_dim = data['obs_dim']
        self.action_dim = data['action_dim']

        # sampler state restore
        if hasattr(self.sampler, "load_state_dict"):
            self.sampler.load_state_dict(data["sampler_state"])

    # ======================================================
    # PICKLE FORMAT
    # ======================================================
    def _dump_pickle(self, path: Path):
        arr = lambda x: x.cpu().numpy()

        state = {
            'obs': arr(self.obs),
            'actions': arr(self.actions),
            'rewards': arr(self.rewards),
            'next_obs': arr(self.next_obs),
            'dones': arr(self.dones),

            'idx': self._idx,
            'full': self._full,
            'history_count': self._history_count,

            'buffer_size': self.buffer_size,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, path: Path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.obs = torch.tensor(data['obs'], device=self.device)
        self.actions = torch.tensor(data['actions'], device=self.device)
        self.rewards = torch.tensor(data['rewards'], device=self.device)
        self.next_obs = torch.tensor(data['next_obs'], device=self.device)
        self.dones = torch.tensor(data['dones'], device=self.device)

        self._idx = data['idx']
        self._full = data['full']
        self._history_count = data.get("history_count", 0)

        self.obs_dim = data['obs_dim']
        self.action_dim = data['action_dim']

    # ==========================================================
    # CSV Dump / Load
    # ==========================================================

    def _dump_csv(self, path: Path):
        """
        Save buffer as CSV.
        Each row = one transition.
        Format:
            obs..., action..., reward, next_obs..., done
        """
        valid = self.size

        with open(path, "w", newline='') as f:
            writer = csv.writer(f)

            # ---- Header ----
            header = []
            header += [f"obs_{i}" for i in range(self.obs.shape[1])]
            header += [f"action_{i}" for i in range(self.actions.shape[1])]
            header += ["reward"]
            header += [f"next_obs_{i}" for i in range(self.next_obs.shape[1])]
            header += ["done"]
            writer.writerow(header)

            # ---- Rows ----
            for i in range(valid):
                row = []
                row += self.obs[i].cpu().tolist()
                row += self.actions[i].cpu().tolist()
                row += [float(self.rewards[i].item())]
                row += self.next_obs[i].cpu().tolist()
                row += [bool(self.dones[i].item())]
                writer.writerow(row)

    def _load_csv(self, path: Path):
        """
        Load transitions from CSV and rebuild buffer.
        Expect CSV columns in same order as dump.
        """
        self.reset()

        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            obs_dim = self.obs.shape[1]
            action_dim = self.actions.shape[1]
            next_obs_dim = self.next_obs.shape[1]

            for row in reader:
                # CSV row is a list of strings
                row = [float(x) if x not in ["True", "False"] else (x == "True") for x in row]

                # Slicing row into components
                o = torch.tensor(row[0:obs_dim], dtype=torch.float32, device=self.device)
                a = torch.tensor(row[obs_dim:obs_dim + action_dim], dtype=torch.float32,
                                 device=self.device)
                r = torch.tensor([row[obs_dim + action_dim]], dtype=torch.float32,
                                 device=self.device)

                no_start = obs_dim + action_dim + 1
                no_end = no_start + next_obs_dim

                no = torch.tensor(row[no_start:no_end], dtype=torch.float32, device=self.device)
                d = torch.tensor([row[-1]], dtype=torch.float32, device=self.device)

                t = Transition(o, a, r, no, d)
                self.add_transition(t)

    # ======================================================
    # Utilities
    # ======================================================
    def get_statistics(self) -> Dict[str, float]:
        valid = self.size

        if valid == 0:
            return {
                "size": 0,
                "capacity": self.buffer_size,
                "utilization": 0.0,
            }

        return {
            "size": valid,
            "capacity": self.buffer_size,
            "history_count": self._history_count,
            "utilization": valid / self.buffer_size,
            "mean_reward": float(self.rewards[:valid].mean()),
            "std_reward": float(self.rewards[:valid].std()),
            "done_ratio": float(self.dones[:valid].mean())
        }

    def __repr__(self):
        return (
            f"SoloReplayBuffer(size={self.size}/{self.buffer_size}, "
            f"ready={self.is_ready}, "
            f"history={self._history_count}, "
            f"device={self.device})"
        )

    @property
    def pointer(self) -> int:
        """Current circular write pointer."""
        return self._idx

    @property
    def size(self) -> int:
        """Return number of valid transitions currently stored."""
        return self.buffer_size if self._full else self._idx

    @property
    def capacity(self) -> int:
        """Return maximum buffer capacity."""
        return self.buffer_size

    @property
    def is_ready(self) -> bool:
        """
        Whether buffer has enough transitions for sampling.
        Returns True when size >= prefill_size.
        """
        return self.size >= self.prefill_size
    
    @property
    def is_prefilled(self) -> bool:
        """Deprecated: use is_ready instead."""
        import warnings
        warnings.warn(
            "is_prefilled is deprecated, use is_ready",
            DeprecationWarning,
            stacklevel=2
        )
        return self.is_ready
    
    @property
    def history_count(self) -> int:
        """Total number of transitions ever written into the buffer (including overwrites)."""
        return self._history_count

    def reset(self):
        """Reset write pointer but DO NOT clear existing data."""
        self._idx = 0
        self._full = False
        self._history_count = 0
        
    def clear(self):
        """
        Clear stored tensor data for replay buffers that define them.
        """
        for attr in ("obs", "actions", "rewards", "next_obs", "dones"):
            if hasattr(self, attr):
                getattr(self, attr).zero_()
        self.reset()
        

    def __len__(self) -> int:
        return self.size
