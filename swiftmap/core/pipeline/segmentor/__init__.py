# Copyright (C) 2024 Carnegie Mellon University
"""Text-promptable semantic segmentation over a reconstruction.

Runs a segmenter (SAM 3) on the frames VGGT reconstructed, reprojects the masks
to 3D via the pixel-aligned world-point map, and clusters the segmented points
into objects (whose centroids NFN turns into GPS waypoints).

Public API:
    BaseSegmenter, register_segmenter, get_segmenter, available_segmenters
    postprocess: generate_seg_scene
"""
from swiftmap.core.pipeline.segmentor.base import BaseSegmenter
from swiftmap.core.pipeline.segmentor.registry import (
    register_segmenter, get_segmenter, available_segmenters, is_registered)
from swiftmap.core.pipeline.segmentor import postprocess
# Importing the backends registers them.
from swiftmap.core.pipeline.segmentor.backends import SAM3Segmenter

__all__ = [
    "BaseSegmenter",
    "register_segmenter",
    "get_segmenter",
    "available_segmenters",
    "is_registered",
    "postprocess",
    "SAM3Segmenter",
]
