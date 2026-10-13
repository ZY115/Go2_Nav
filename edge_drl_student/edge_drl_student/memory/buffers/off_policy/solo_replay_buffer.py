# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
from pathlib import Path
from edge_drl_student.utils.utils import ActionMode
from typing import Dict, Generator, Tuple, Optional, Union
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.samplers.uniform_sampler import UniformSampler
from edge_drl_student.memory.buffers import Transition, TransitionBatch


class SoloReplayBuffer(BaseBuffer):
    """
    SoloReplayBuffer
    ----------------
    A united replay buffer implementation built on top of the base class.
    """

    def __init__(self, 
                 buffer_size: int, 
                 prefill_size: int,
                 sampler: Optional[UniformSampler] = None, 
                 obs_dim: int = 12,
                 action_dim: int = 6,
                 name: str = "SoloReplayBuffer",
                 device: Optional[str] = 'cpu'):
        if sampler is None:
            sampler = UniformSampler(buffer_size, random_seed=42, device=device)
            
        super().__init__(buffer_size, prefill_size, sampler, name, device=device)

        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=self.device)
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def add_transition(self, t: Transition, action_mode: ActionMode = ActionMode.UNCERTAIN):
        """
        Add one transition into the buffer.
        Typically called at every environment step.
        """
        idx = self._idx

        self.obs[idx] = t.obs
        self.actions[idx] = t.action
        self.rewards[idx] = t.reward
        self.next_obs[idx] = t.next_obs
        self.dones[idx] = t.done

        self._idx = (self._idx + 1) % self.buffer_size  # Circular buffer
        self._history_count += 1                        # Total transitions added
        
        if self._idx == 0 and not self._full:
            self._full = True

    def _get_samples(self, indices: torch.Tensor) -> TransitionBatch:
        """
        Given sampled indices, return the corresponding transitions.
        """
        return TransitionBatch(
            obs=self.obs[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_obs=self.next_obs[indices],
            done=self.dones[indices]
        )
    
    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a batch of transitions using the attached sampler.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            A batch of transitions obtained via `_get_samples()`.
        """
        valid_size = self.size
        if valid_size <= 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if valid_size > self.buffer_size:
            raise ValueError(
                f"valid_size ({valid_size}) exceeds buffer_size ({self.buffer_size})."
            )
        if not self.is_ready:
            raise RuntimeError(
                f"Buffer not prefilled: current size is {self.size}, "
                f"required prefill size is {self.prefill_size}."
            )

        # Delegate sampling to sampler
        sample_indices = self.sampler.fetch(batch_size, valid_size)
        return self._get_samples(sample_indices)
    


if __name__ == "__main__":
    # ======================================================
    # 1. create buffer
    # ======================================================
    buf = SoloReplayBuffer(
        buffer_size=10,
        prefill_size=3,
        obs_dim=4,
        action_dim=2,
        device="cpu"
    )

    print("Initial buffer:", buf)

    # ======================================================
    # 2. write in transition
    # ======================================================
    for i in range(12):
        t = Transition(
            obs=torch.tensor([i, i+1, i+2, i+3], dtype=torch.float32),
            action=torch.tensor([i * 0.1, i * 0.2]),
            reward=torch.tensor([i * 1.0]),
            next_obs=torch.tensor([i+0.1, i+1.1, i+2.1, i+3.1]),
            done=torch.tensor([i % 2 == 0], dtype=torch.float32)
        )
        buf.add_transition(t)

    print("\nAfter adding 12 transitions:")
    print(buf.get_statistics())

    # ======================================================
    # 3. Dump to 3 formats
    # ======================================================
    out_dir = Path("./buffer_test_output")
    out_dir.mkdir(exist_ok=True)

    path_pt = out_dir / "buffer.pt"
    path_pkl = out_dir / "buffer.pkl"
    path_csv = out_dir / "buffer.csv"

    buf.dump(path_pt, format="torch")
    buf.dump(path_pkl, format="pickle")
    buf.dump(path_csv, format="csv")
    
    # ======================================================
    # 4. Load back into new buffers
    # ======================================================
    buf_pt = SoloReplayBuffer(10, 3, obs_dim=4, action_dim=2)
    buf_pt.load(path_pt)

    buf_pkl = SoloReplayBuffer(10, 3, obs_dim=4, action_dim=2)
    buf_pkl.load(path_pkl)

    buf_csv = SoloReplayBuffer(10, 3, obs_dim=4, action_dim=2)
    buf_csv.load(path_csv)

    # ======================================================
    # 5. Compare data to verify correctness
    # ======================================================
    assert torch.allclose(buf.obs, buf_pt.obs)
    assert torch.allclose(buf.actions, buf_pt.actions)
    assert torch.allclose(buf.rewards, buf_pt.rewards)
    assert torch.allclose(buf.next_obs, buf_pt.next_obs)
    assert torch.allclose(buf.dones, buf_pt.dones)

    assert torch.allclose(buf.obs, buf_pkl.obs)
    assert torch.allclose(buf.obs, buf_csv.obs)

    print("\nAll formats loaded correctly!")
    print("✓ torch OK")
    print("✓ pickle OK")
    print("✓ csv OK")

    print("\nPT statistics:", buf_pt.get_statistics())
    print("PKL statistics:", buf_pkl.get_statistics())
    print("CSV statistics:", buf_csv.get_statistics())
    
    print("\n==== TEST PASSED ====")