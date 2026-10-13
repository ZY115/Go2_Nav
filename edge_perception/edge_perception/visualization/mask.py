#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class MaskBlob:
    mask: np.ndarray  # bool [H, W]
    score: float = 1.0
    instance_id: Optional[int] = None