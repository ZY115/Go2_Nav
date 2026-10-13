# #!/bin/bash
# # build_sam_engine.sh
# # TensorRT engine builder for SAM ViT-B
# # For images of size 1280x720, with 5 points per image, FP16 engine

# # Set ONNX model and output engine file
# ONNX_MODEL="onnx_export/sam_vit_b.onnx"
# ENGINE_NAME="onnx_export/sam_vit_b_fp16.engine"

# # GPU workspace size in MB
# WORKSPACE=8192

# # Build TensorRT engine
# /usr/src/tensorrt/bin/trtexec \
#     --onnx=${ONNX_MODEL} \
#     --saveEngine=${ENGINE_NAME} \
#     --fp16 \
#     --workspace=${WORKSPACE} \
#     --minShapes=image_embeddings:1x256x64x64,point_coords:1x5x2,point_labels:1x5,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1x2 \
#     --optShapes=image_embeddings:1x256x64x64,point_coords:1x5x2,point_labels:1x5,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1x2 \
#     --maxShapes=image_embeddings:1x256x64x64,point_coords:1x5x2,point_labels:1x5,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1x2 \
#     --skipInference

# echo "=== TensorRT engine build finished ==="
# echo "Engine saved as: ${ENGINE_NAME}"

/usr/src/tensorrt/bin/trtexec --verbose --onnx=onnx_export/sam_vit_b.onnx --saveEngine=onnx_export/sam_vit_b_fp16.trt --fp16
