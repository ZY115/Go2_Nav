#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import cv2
import numpy as np
from typing import List

from dataclasses import dataclass
from typing import Optional

class MaskVisualizer:

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def apply_mask(self, image_rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, inplace: bool = True) -> np.ndarray:
        """
        Apply the given mask to the image with specified color and transparency.

        Parameters:
            image_rgb (np.ndarray): Original RGB image
            mask (np.ndarray): Boolean mask, [H, W]
            color (np.ndarray or list): Fill and contour color
            inplace (bool): Whether to modify the image in place

        Returns:
            np.ndarray: Image with the mask applied
        """
        if not inplace:
            image_rgb = image_rgb.copy()

        # Fill the mask area
        image_rgb[mask] = image_rgb[mask] * (1 - self.alpha) + np.array(color) * self.alpha

        # Draw contours
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_rgb, contours, -1, color.tolist(), 2)

        return image_rgb