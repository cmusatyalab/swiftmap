# Copyright (C) 2024 Carnegie Mellon University
"""Text-promptable semantic segmentation over a site.

Runs a segmenter (SAM 3) on the frames every map in the site was reconstructed from,
reprojects the masks to 3D via the pixel-aligned world-point map, and carries them
into the site's shared ENU frame. The Site owns the resulting targets and exports
them as seg_site.glb.

Public API:
    BaseSegmenter, register_segmenter, get_segmenter, available_segmenters
"""
from swiftmap.core.pipeline.segmentor.base import BaseSegmenter
from swiftmap.core.pipeline.segmentor.registry import (
    register_segmenter, get_segmenter, available_segmenters, is_registered)
# Importing the backends registers them.
from swiftmap.core.pipeline.segmentor.backends import SAM3Segmenter

__all__ = [
    "BaseSegmenter",
    "register_segmenter",
    "get_segmenter",
    "available_segmenters",
    "is_registered",
    "SAM3Segmenter",
]
