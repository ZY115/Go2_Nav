#!/bin/bash

ARCH=$(uname -m)

case "${ARCH}" in
    x86_64)
        PLATFORM="x86_64"
        GPU_FLAG="--gpus all"
        ;;
    aarch64|arm64)
        if [ -f /etc/nv_tegra_release ]; then
            PLATFORM="jetson"
            GPU_FLAG="--runtime nvidia"
        else
            PLATFORM="arm64"
            GPU_FLAG=""
        fi
        ;;
    *)
        echo "[ERROR] Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

# Check if DISPLAY is set, if not, default to :0
if [ -z "$DISPLAY" ]; then
    echo "[INFO] DISPLAY not set, defaulting to :0"
    DISPLAY=:0
fi

echo "[INFO] Using DISPLAY=$DISPLAY"
echo "[INFO] Using PLATFORM=$PLATFORM"

docker run -it --rm \
  $GPU_FLAG \
  --net host \
  --pid host \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  edge-brain-go2:${PLATFORM}
