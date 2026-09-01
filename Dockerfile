# Copyright (C) 2024 Carnegie Mellon University
# SwiftMap headless mapping server (VGGT / VGGT-Omega + SAM 3).
#
# Receives frame+GPS pairs over TCP (default 43322) and auto-runs the pipeline at
# the keyframe cap, exporting results to /app/output (mount a host volume there).
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

LABEL org.opencontainers.image.vendor="CMU Satyalab" \
      org.opencontainers.image.description="SwiftMap headless mapping server"

ARG DEBIAN_FRONTEND=noninteractive

# System deps: Python 3.10 (matches requires-python), git (for the model packages),
# build tools (pycocotools builds from source), OpenGL/glib (OpenCV/trimesh).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git build-essential \
        libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Python dependencies. SAM 3 still imports pkg_resources, so setuptools stays <81.
# Model backbones come from their pinned upstream repos (the [vggt]/[vggt-omega]/[sam3]
# extras); keep these SHAs in sync with pyproject.toml.
RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<81" wheel && \
    python -m pip install --no-cache-dir \
        torch==2.3.1 torchvision==0.18.1 "numpy<2" Pillow einops safetensors huggingface_hub \
        opencv-python-headless trimesh scipy matplotlib requests pymap3d onnxruntime \
        timm ftfy regex iopath pycocotools psutil "gradio>=5.49" && \
    python -m pip install --no-cache-dir \
        "vggt @ git+https://github.com/facebookresearch/vggt.git@c3953fabdcc66fe52b578f27c59516d567416953" \
        "vggt-omega @ git+https://github.com/facebookresearch/vggt-omega.git@39a0cb8" \
        "sam3 @ git+https://github.com/facebookresearch/sam3.git"

# App source.
COPY . .

# Model weights are supplied at run time (not baked into the image):
#   * VGGT + SAM 3 download from HuggingFace on first use to the default caches
#     (/root/.cache/torch and /root/.cache/huggingface) — mount a volume at
#     /root/.cache to persist them across restarts (download-once).
#   * VGGT-Omega has no public download — mount its checkpoint at /app/checkpoints
#     and point VGGT_OMEGA_CHECKPOINT at it.
ENV SWIFTMAP_OUTPUT_DIR=/app/output \
    PYTHONUNBUFFERED=1

EXPOSE 43322 7866
# 43322 = TCP frame+GPS ingest, 7866 = passive results viewer (Gradio)

CMD ["python", "launch_server.py"]
