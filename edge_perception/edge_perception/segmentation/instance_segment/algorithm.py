#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import torch
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from ultralytics.engine.results import Results

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from edge_perception.visualization.mask import MaskBlob

class SegmentModel(ABC):
    @abstractmethod
    def infer(self, image_rgb: np.ndarray) -> List[MaskBlob]:
        pass

class ModelGenerator:
    model: SegmentModel
    def __init__(self, algorithm: str, **kwargs):
        algorithm = algorithm.upper()
        if algorithm == 'SAM':
            self.model = SAMWrapper(**kwargs)
        elif algorithm == 'YOLO':
            self.model = YOLOWrapper(**kwargs)
        else:
            raise ValueError(f'Unsupported algorithm: {algorithm}')
        
    
class SAMWrapper(SegmentModel):
    def __init__(self, model_path, model_type, device, **kwargs):
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        self.device = device

        self.generator = SamAutomaticMaskGenerator(sam, **kwargs)

    def infer(self, image_rgb: np.ndarray) -> list[MaskBlob]:
        sam_masks = self.generator.generate(image_rgb)

        return [
            MaskBlob(
                mask=m["segmentation"],
                score=m.get("predicted_iou", 1.0),
                instance_id=i
            )
            for i, m in enumerate(sam_masks)
        ]

class YOLOWrapper(SegmentModel):
    def __init__(self, model_path, device, **kwargs):
        self.model = YOLO(model_path)
        self.model.to(device=device)
        self.device = device
        # Optional thresholds
        self.conf = kwargs.get('confidence_threshold')
        self.iou = kwargs.get('iou_threshold')

    def infer(self, image_rgb: np.ndarray) -> list[MaskBlob]:
        # Ultralytics YOLO handles preprocessing internally - pass numpy array directly
        infer_kwargs = {"verbose": False}
        # if self.conf is not None:
        #     infer_kwargs["conf"] = float(self.conf)
        # if self.iou is not None:
        #     infer_kwargs["iou"] = float(self.iou)
        results = self.model(image_rgb, **infer_kwargs)

        res: Results = results[0]
        if res.masks is None:
            return []

        instance_masks = []
        masks: torch.Tensor = res.masks.data  # [N, H, W] (torch)
        scores = res.masks.conf  # tensor [N]
        
        for i in range(masks.shape[0]):
            instance_masks.append(
                MaskBlob(
                    mask=masks[i].cpu().numpy().astype(bool),
                    score=float(scores[i].cpu().numpy()),
                    instance_id=i
                )
            )
        return instance_masks
