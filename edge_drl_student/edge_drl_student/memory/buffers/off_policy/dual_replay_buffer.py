# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Callable, Dict, Generator, Tuple, Optional, Union

from edge_drl_student.memory.buffers import SoloReplayBuffer
from edge_drl_student.memory.buffers.base_buffer import BaseBuffer
from edge_drl_student.memory.samplers.uniform_sampler import UniformSampler
from edge_drl_student.memory.buffers import Transition, TransitionBatch
from edge_drl_student.utils.utils import ActionMode, safety_value

ENVELOPE_P = np.array([[100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0.25, -0, 0, -0, -0, -0, -0, -0, -0],
                       [0, -0, 0.25, 0, -0, -0, -0, -0, -0, -0],
                       [0, 0, 0, 0.6944, -0, 0, 0, 0, 0, 0],
                       [0, -0, -0, -0, 11.1111, -0, -0, -0, -0, -0],
                       [0, -0, -0, 0, -0, 0.25, -0, -0, -0, -0],
                       [0, -0, -0, 0, -0, -0, 0.25, -0, -0, -0],
                       [0, -0, -0, 0, -0, -0, -0, 0.25, -0, -0],
                       [0, -0, -0, 0, -0, -0, -0, -0, 0.25, -0],
                       [0, -0, -0, 0, -0, -0, -0, -0, -0, 4]])

class DualReplayBuffer(BaseBuffer):
    def __init__(self, 
                 student_buf_size: int, 
                 student_prefill_size: int,
                 teacher_buf_size: int, 
                 teacher_prefill_size: int,
                 sampler: Optional[UniformSampler] = None,
                 obs_dim: int = 12,
                 action_dim: int = 6,
                 device: str = "cpu",
                 ) -> None:

        if sampler is None:
            sampler = UniformSampler(buffer_size=max(student_buf_size, teacher_buf_size), 
                                     random_seed=42, device=device)

        self.student_buf_size = student_buf_size       
        self.teacher_buf_size = teacher_buf_size        
        self.device = device

        # Self-learning DRL-Student
        self.stu_buf = SoloReplayBuffer(buffer_size=self.student_buf_size, 
                                        prefill_size=student_prefill_size,
                                        obs_dim=obs_dim, 
                                        action_dim=action_dim,
                                        sampler=sampler,
                                        name="DRL-Student Replay Buffer",
                                        device=self.device)

        # Teaching-to-learn PHY-Teacher
        self.tea_buf = SoloReplayBuffer(buffer_size=self.teacher_buf_size, 
                                        prefill_size=teacher_prefill_size,
                                        obs_dim=obs_dim, 
                                        action_dim=action_dim,
                                        sampler=sampler,
                                        name="PHY-Teacher Replay Buffer",
                                        device=self.device)

        self.last_data_type = ActionMode.TEACHER


    @property
    def stu_buf_size(self) -> int:
        return self.stu_buf.size

    @property
    def tea_buf_size(self) -> int:
        return self.tea_buf.size

    def add_transition(self, transition: Transition, action_mode: ActionMode) -> None:
        """Appends a dataset of transitions to the storage.

        Args:
            dataset (Dataset): The dataset of transitions.
        """

        if action_mode == ActionMode.TEACHER:
            self.tea_buf.add_transition(t=transition)
            self.last_data_type = ActionMode.TEACHER

        elif action_mode == ActionMode.STUDENT:
            self.stu_buf.add_transition(t=transition)
            self.last_data_type = ActionMode.STUDENT

        else:
            raise RuntimeError(f"Unrecognized action mode: {action_mode} for this transition.")

    def dump(self, path: Union[str, Path], format: str = 'torch', compress: bool = False):
        """Save both student and teacher replay buffers to a directory."""
        dump_dir = Path(path)
        dump_dir.mkdir(parents=True, exist_ok=True)

        self.stu_buf.dump(dump_dir / "stu_buffer.pt", format='torch', compress=compress)
        self.tea_buf.dump(dump_dir / "tea_buffer.pt", format='torch', compress=compress)

    def load(self,
             path: Union[str, Path],
             format: str = 'auto',
             device: Optional[str] = None):
        """Load student and teacher replay buffers from a directory."""
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(load_dir)
        if not load_dir.is_dir():
            raise ValueError(f"DualReplayBuffer expects a directory path, got: {load_dir}")

        stu_path = load_dir / "stu_buffer.pt"
        tea_path = load_dir / "tea_buffer.pt"
        if not stu_path.exists() or not tea_path.exists():
            raise FileNotFoundError(
                f"Missing buffer snapshot files under {load_dir}. "
                f"Expected: {stu_path.name}, {tea_path.name}"
            )

        self.stu_buf.load(stu_path, format='auto', device=device)
        self.tea_buf.load(tea_path, format='auto', device=device)


    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a batch of transitions using the attached sampler.

        Args:
            batch_size (int): The size of the batch to sample.
        Returns:
            A batch of sampled transitions.
        """
        
        # Invalid batch size check
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        elif batch_size > self.stu_buf.size + self.tea_buf.size:
            raise ValueError(
                f"Requested batch size {L} exceeds total stored transitions "
                f"({self.stu_buf.size + self.tea_buf.size})."
            )

        L = batch_size
        if self.last_data_type == ActionMode.STUDENT:
            stu_idx = self.stu_buf.pointer - 1
            self.boundary_obs = self.stu_buf.obs[stu_idx]
            
            # Boundary obs might contain vision features
            if self.boundary_obs.size(0) > 12:
                boundary_state = self.boundary_obs[:12]
            else:
                boundary_state = self.boundary_obs

        elif self.last_data_type == ActionMode.TEACHER:
            tea_idx = self.tea_buf.pointer - 1
            self.boundary_obs = self.tea_buf.obs[tea_idx]
            
            # Boundary obs might contain vision features
            if self.boundary_obs.size(0) > 12:
                boundary_state = self.boundary_obs[:12]
            else:
                boundary_state = self.boundary_obs
        else:
            raise RuntimeError(f"Unknown action type pointer: {self.last_data_type}")

        Vs = safety_value(boundary_state[2:].cpu().numpy(), ENVELOPE_P) * 0.01

        # Batch size for PHY-Teacher Buffer
        tea_batch_size = max(min(L - 1, int(L * Vs)), 1)
        
        tea_batch_size = 50

        # Batch size for DRL-Student Buffer
        stu_batch_size = L - tea_batch_size
        if stu_batch_size < 0:
            raise RuntimeError("Negative student batch size encountered.")
        
        stu_batch = self.stu_buf.sample(stu_batch_size) if stu_batch_size > 0 else None
        tea_batch = self.tea_buf.sample(tea_batch_size) if tea_batch_size > 0 else None

        stu_dict = asdict(stu_batch) if stu_batch is not None else None
        tea_dict = asdict(tea_batch) if tea_batch is not None else None

        print(f"stu_batch_size: {stu_batch_size}, tea_batch_size: {tea_batch_size}")
        # print(f"stu_batch: {stu_batch}, tea_batch: {tea_batch}")
        # print(f"stu_dict: {stu_dict}, tea_dict: {tea_dict}")

        # Merge two batches safely
        merged_dict = {}
        keys = stu_dict.keys() if stu_dict is not None else tea_dict.keys()
        
        for k in keys:
            # Get non-None values
            vals = [b[k] for b in (stu_dict, tea_dict) if b is not None and b[k] is not None]

            # Case 1: Both batches have None for this key, e.g., info
            if len(vals) == 0:
                merged_dict[k] = None
                continue

            # Case 2: If not a Tensor (e.g., info has a value)
            if not isinstance(vals[0], torch.Tensor):
                merged_dict[k] = vals[0]
                continue

            # Case 3: Tensor → concat
            merged_dict[k] = torch.cat(vals, dim=0)

        return TransitionBatch(**merged_dict)


if __name__ == "__main__":
    # ======================================================
    # 1. create buffer
    # ======================================================
    dual_buf = DualReplayBuffer(
        student_buf_size=10,
        student_prefill_size=3,
        teacher_buf_size=10,
        teacher_prefill_size=3,
        obs_dim=12,
        action_dim=6,
        device="cuda:0"
    )

    # ======================================================
    # 2. write in transition
    # ======================================================
    for i in range(24):
        t = Transition(
            obs=torch.tensor([i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9, i+10, i+11], dtype=torch.float32),
            action=torch.tensor([i * 0.1, i * 0.2, i * 0.3, i * 0.4, i * 0.5, i * 0.6], dtype=torch.float32),
            reward=torch.tensor([i * 1.0], dtype=torch.float32),
            next_obs=torch.tensor([i+0.1, i+1.1, i+2.1, i+3.1, i+4.1, i+5.1, i+6.1, i+7.1, i+8.1, i+9.1, i+10.1, i+11.1], dtype=torch.float32),
            done=torch.tensor([i % 2 == 0], dtype=torch.float32)
        )
        dual_buf.add_transition(t, action_mode=ActionMode.STUDENT if i % 2 == 0 else ActionMode.TEACHER)
        
    print(f"Student Buffer Size: {dual_buf.stu_buf_size}")
    print(f"Teacher Buffer Size: {dual_buf.tea_buf_size}")
    
    sampled_batch = dual_buf.sample(batch_size=5)
    print(f"Sampled Batch: {sampled_batch}")
    
    dual_buf.stu_buf.dump(path="./stu_buffer_dump.txt", format='csv')
    dual_buf.tea_buf.dump(path="./tea_buffer_dump.txt", format='csv')