#!/bin/bash
set -e

# ======================================================
# edge-brain-go2 Docker Build Script
# - Auto arch detection (x86 / jetson)
# - Auto Dockerfile & tag selection
# - Auto ssh-agent handling
# ======================================================

# -------- Detect Architecture --------
ARCH=$(uname -m)

case "${ARCH}" in
    x86_64)
        PLATFORM="x86_64"
        ;;
    aarch64|arm64)
        # Jetson
        if [ -f /etc/nv_tegra_release ]; then
            PLATFORM="jetson"
        else
            # Standard ARM64
            PLATFORM="arm64"
        fi
        ;;
    *)
        echo "[ERROR] Unsupported architecture: ${ARCH}"
        exit 1
        ;;
esac

echo "[INFO] Detected architecture: ${ARCH} -> ${PLATFORM}"

# -------- Assigned CPU Jobs --------
MAKE_JOBS=$(nproc)
echo "[INFO] Using MAKE_JOBS=${MAKE_JOBS}"

# -------- Image Names --------
BASE_IMAGE_NAME=edge-brain-base
BASE_IMAGE_TAG=${PLATFORM}

APP_IMAGE_NAME=edge-brain-go2
APP_IMAGE_TAG=${PLATFORM}

# -------- Dockerfiles --------
BASE_DOCKERFILE=../dockerfile/base_${PLATFORM}.Dockerfile
APP_DOCKERFILE=../dockerfile/app.Dockerfile

# -------- Enable BuildKit --------
export DOCKER_BUILDKIT=1

# -------- Start SSH agent --------
SSH_KEY="$HOME/.ssh/id_rsa"
SSH_OPTION=""

if [ -f "$SSH_KEY" ]; then
    echo "[INFO] Starting ssh-agent and adding key $SSH_KEY"
    eval "$(ssh-agent -s)"
    ssh-add "$SSH_KEY"
    echo "[DEBUG] SSH_AUTH_SOCK=$SSH_AUTH_SOCK"

    if [ -n "$SSH_AUTH_SOCK" ] && [ -S "$SSH_AUTH_SOCK" ]; then
        chmod 666 "$SSH_AUTH_SOCK"
        echo "[DEBUG] Fixed SSH socket permissions"
    fi

    SSH_OPTION="--ssh default"
else
    echo "[INFO] No SSH key found at $SSH_KEY, building WITHOUT SSH"
fi

# -------- Sanity Check --------
if [ ! -f "${BASE_DOCKERFILE}" ]; then
    echo "[ERROR] Base Dockerfile not found: ${BASE_DOCKERFILE}"
    exit 1
fi

if [ ! -f "${APP_DOCKERFILE}" ]; then
    echo "[ERROR] App Dockerfile not found: ${APP_DOCKERFILE}"
    exit 1
fi

# ======================================================
# Build Base Image
# ======================================================
echo "=============================="
echo "Building Base Image:"
echo "  ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
echo "  Dockerfile: ${BASE_DOCKERFILE}"
echo "=============================="

docker build \
    --build-arg MAKE_JOBS=${MAKE_JOBS} \
    -f ${BASE_DOCKERFILE} \
    -t ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} .

# ======================================================
# Build App Image
# ======================================================
echo "=============================="
echo "Building App Image:"
echo "  ${APP_IMAGE_NAME}:${APP_IMAGE_TAG}"
echo "  Dockerfile: ${APP_DOCKERFILE}"
echo "=============================="

docker build \
    ${SSH_OPTION} \
    -f "${APP_DOCKERFILE}" \
    --build-arg MAKE_JOBS=${MAKE_JOBS} \
    --build-arg PLATFORM=${PLATFORM} \
    --build-arg BASE_IMAGE=${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} \
    -t "${APP_IMAGE_NAME}:${APP_IMAGE_TAG}" .

# ======================================================
# Done
# ======================================================
echo "=============================="
echo "Docker Build Completed ✅"
echo "Platform:   ${PLATFORM}"
echo "Base Image: ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}"
echo "App Image:  ${APP_IMAGE_NAME}:${APP_IMAGE_TAG}"
echo "=============================="
