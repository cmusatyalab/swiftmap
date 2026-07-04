# Copyright (C) 2024 Carnegie Mellon University

"""
SwiftMap

A real-time drone mapping system that:
1. Receives drone images via TCP stream
2. Performs optical flow-based keyframe selection  
3. Runs VGGT inference on selected keyframes
4. Provides dual 3D visualization (reconstruction + confidence mapping)

Key Components:
- TCP Server: Receives images from drone clients on port 43322
- Keyframe Selector: Uses optical flow to identify important frames
- VGGT Mapper: Performs 3D reconstruction inference
- Gradio Interface: Dual viewer for 3D models and confidence maps

Usage:
    python launch_mapping.py
"""

__version__ = "1.0.0"

# Main system components
from swiftmap.core.session import MappingSession
from swiftmap.core.tcp_server import MappingTCPServer
from swiftmap.core.keyframe_selector import KeyframeSelector
from swiftmap.core.mapper import (
    BaseMapper, VGGTMapper, VGGTOmegaMapper, get_mapper, available_mappers)
from swiftmap.frontend.gradio_interface import MappingGradioInterface

# Keyframe selection helper
from swiftmap.core.keyframe_selector import FrameTracker

__all__ = [
    "MappingSession",
    "MappingTCPServer",
    "KeyframeSelector",
    "BaseMapper",
    "VGGTMapper",
    "VGGTOmegaMapper",
    "get_mapper",
    "available_mappers",
    "MappingGradioInterface",
    "FrameTracker"
]