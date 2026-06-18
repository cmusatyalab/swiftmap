# Copyright (C) 2024 Carnegie Mellon University
"""
SwiftMap frontend.

The Gradio web interface — the front-end orchestration layer that drives the core
(keyframe selection, VGGT mapping, NFN planning) and renders the 3D viewers. Not
part of the domain core.
"""

from swiftmap.frontend.gradio_interface import MappingGradioInterface

__all__ = ["MappingGradioInterface"]
