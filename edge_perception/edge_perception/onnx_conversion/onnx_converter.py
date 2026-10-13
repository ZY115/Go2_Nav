#!/usr/bin/env python3
# ============================================================================
# Copyright (c) 2025-2026 Yihao Cai - All rights reserved
# SPDX-License-Identifier: Apache-2.0
# ============================================================================

import os
import torch
import warnings
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
from ament_index_python.packages import get_package_share_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_CKPT_DIR = os.path.join(BASE_DIR, "raw_checkpoints")
ONNX_EXPORT_DIR = os.path.join(BASE_DIR, "onnx_export")

# Model checkpoint and ONNX export path
model_type = "vit_b"
ckpt_name = "sam_vit_b_01ec64.pth"
exported_onnx_name = "sam_vit_b.onnx"

checkpoint = os.path.join(RAW_CKPT_DIR, ckpt_name)
onnx_model_path = os.path.join(ONNX_EXPORT_DIR, exported_onnx_name)
    
# ONNX export should run on CPU to avoid device mismatches in SAM internals
DEVICE = "cpu"
DTYPE = torch.float32  

# ---------------- Load SAM ----------------
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=DEVICE)
sam.eval()

onnx_model = SamOnnxModel(
    sam,
    return_single_mask=True
)

# ---------------- Fixed dimension definitions (key) ----------------
embed_dim = sam.prompt_encoder.embed_dim                 # 256
embed_size = sam.prompt_encoder.image_embedding_size     # [64, 64]
mask_input_size = [4 * x for x in embed_size]            # [256, 256]

NUM_POINTS = 5   # Fixed maximum number of points (runtime ≤ this)

# ----------------  dummy inputs (determine ONNX shape) ----------------
dummy_inputs = {
    "image_embeddings": torch.randn(
        1, embed_dim, embed_size[0], embed_size[1],
        device=DEVICE, dtype=DTYPE
    ),
    "point_coords": torch.randn(
        1, NUM_POINTS, 2,
        device=DEVICE, dtype=DTYPE
    ),
    "point_labels": torch.zeros(
        1, NUM_POINTS,
            device=DEVICE, dtype=torch.int32
    ),
    "mask_input": torch.zeros(
        1, 1, mask_input_size[0], mask_input_size[1],
        device=DEVICE, dtype=DTYPE
    ),
    "has_mask_input": torch.zeros(
        1, device=DEVICE, dtype=DTYPE
    ),
    "orig_im_size": torch.tensor(
           [720, 1280],   # Original image size
           device=DEVICE, dtype=torch.int32
    ),
}

# ---------------- Only keep necessary dynamic dimensions ----------------
dynamic_axes = {
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}

output_names = [
    "masks",
    "iou_predictions",
    "low_res_masks"
]

# ---------------- 导出 ONNX ----------------
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        onnx_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=list(dummy_inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

print(f"[OK] ONNX exported to {onnx_model_path}")
