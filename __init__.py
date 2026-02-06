# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VGGT Mapping System

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
    python vggt_mapping/launch_mapping.py
"""

__version__ = "1.0.0"
__author__ = "Meta AI"

# Main system components
from vggt_mapping.core.tcp_server import MappingTCPServer
from vggt_mapping.core.keyframe_selector import KeyframeSelector
from vggt_mapping.core.vggt_mapper import VGGTMapper
from vggt_mapping.core.gradio_interface import MappingGradioInterface

# Utilities
from vggt_mapping.utils.frame_tracker import FrameTracker

__all__ = [
    "MappingTCPServer",
    "KeyframeSelector", 
    "VGGTMapper",
    "MappingGradioInterface",
    "FrameTracker"
]