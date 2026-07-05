# Copyright (C) 2024 Carnegie Mellon University
"""Text-promptable semantic segmentation over a reconstruction.

Runs a segmenter (SAM 3) on the frames VGGT reconstructed, reprojects the masks
to 3D via the pixel-aligned world-point map, and clusters the segmented points
into objects (whose centroids NFN turns into GPS waypoints).

Public API:
    BaseSegmenter, register_segmenter, get_segmenter, available_segmenters
    lift: frame_images, masks_to_points, export_highlight_glb, cluster_objects
"""
from swiftmap.core.semantic.base import BaseSegmenter
from swiftmap.core.semantic.registry import (
    register_segmenter, get_segmenter, available_segmenters, is_registered)
from swiftmap.core.semantic import lift
# Importing the backends registers them.
from swiftmap.core.semantic.backends import SAM3Segmenter

__all__ = [
    "BaseSegmenter",
    "register_segmenter",
    "get_segmenter",
    "available_segmenters",
    "is_registered",
    "lift",
    "SAM3Segmenter",
]
