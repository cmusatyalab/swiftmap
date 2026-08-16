# Copyright (C) 2024 Carnegie Mellon University
"""Backbone adapters. Importing this package registers each backbone."""
from swiftmap.core.pipeline.reconstructor.backends.vggt import VGGTMapper
from swiftmap.core.pipeline.reconstructor.backends.vggt_omega import VGGTOmegaMapper

__all__ = ["VGGTMapper", "VGGTOmegaMapper"]
