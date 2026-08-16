# Copyright (C) 2024 Carnegie Mellon University

"""
SwiftMap default parameters, service ports, and model asset URLs.

Centralizes the values that were previously hardcoded in several modules so there
is a single place to tune them. Wire-protocol constants live in ``protocol.py``.
"""

import os

# --- Keyframe selection -----------------------------------------------------
# Minimum mean optical-flow disparity (px) for a frame to be kept as a keyframe.
DEFAULT_MIN_DISPARITY = 40.0
# After selection, the keyframe set sent to VGGT is capped to this many (0 = no cap).
DEFAULT_MAX_KEYFRAMES = 70

# --- Reconstruction / confidence -------------------------------------------
# Confidence threshold (percentile, %) used when filtering the point cloud.
DEFAULT_CONF_THRESHOLD = 60.0

# --- Next-Flight Navigation (NFN) ------------------------------------------
# Percentile band that marks "to-improve" (low-but-not-lowest confidence) regions.
NFN_LOW_PERCENTILE = 60.0
NFN_HIGH_PERCENTILE = 80.0

# --- Service ports ----------------------------------------------------------
GUI_PORT = 7866      # Gradio web interface
NFN_VISER_PORT = 7867  # Viser NFN viewer (opened on demand)

# --- Model asset URLs -------------------------------------------------------
VGGT_MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
SKYSEG_ONNX_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
SKYSEG_ONNX_PATH = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "swiftmap", "skyseg.onnx")

# --- VGGT-Omega -------------------------------------------------------------
# Omega ships no weights on a hub; it loads from a local checkpoint (.pt) via
# torch.load + load_state_dict. Point this at your checkpoint (env override
# wins) and match the resolution to the checkpoint you use (512 -> ...512).
VGGT_OMEGA_CHECKPOINT = os.environ.get(
    "VGGT_OMEGA_CHECKPOINT",
    "/home/ubuntu/xianglic/vggt-omega/checkpoints/vggt_omega_1b_512-002.pt")
VGGT_OMEGA_IMAGE_RESOLUTION = int(os.environ.get("VGGT_OMEGA_IMAGE_RESOLUTION", "512"))

# --- Semantic segmentation (SAM 3) -----------------------------------------
# Text-promptable segmentation over the frames VGGT reconstructed, reprojected
# to 3D via the (pixel-aligned) world-point map. Weights: point SAM3_CHECKPOINT
# at a local .pt (env override wins); if empty/missing, the SAM 3 builder falls
# back to downloading facebook/sam3 from the HuggingFace Hub.
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", "")  # "" -> download from HF
# BPE tokenizer vocab shipped in the SAM 3 repo assets (needed by the text encoder).
SAM3_BPE_PATH = os.environ.get(
    "SAM3_BPE_PATH",
    "/home/ubuntu/xianglic/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
# Minimum SAM 3 instance confidence to keep a mask.
SAM3_CONF_THRESHOLD = float(os.environ.get("SAM3_CONF_THRESHOLD", "0.5"))
# Default segmentation backend (registered in swiftmap.core.pipeline.segmentor).
DEFAULT_SEGMENTER = os.environ.get("SWIFTMAP_SEGMENTER", "sam3")

# --- Segmented-object clustering (v1: spatial) -----------------------------
# Segmented 3D points are clustered into objects by spatial density (DBSCAN-like
# over a KD-tree). eps is expressed as a fraction of the scene diagonal so it is
# scale-invariant across reconstructions.
SEG_CLUSTER_EPS_FRACTION = float(os.environ.get("SEG_CLUSTER_EPS_FRACTION", "0.02"))
SEG_CLUSTER_MIN_POINTS = int(os.environ.get("SEG_CLUSTER_MIN_POINTS", "20"))
