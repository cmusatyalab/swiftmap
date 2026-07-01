# Copyright (C) 2024 Carnegie Mellon University

"""
SwiftMap default parameters, service ports, and model asset URLs.

Centralizes the values that were previously hardcoded in several modules so there
is a single place to tune them. Wire-protocol constants live in ``protocol.py``.
"""

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
SKYSEG_ONNX_FILENAME = "skyseg.onnx"
