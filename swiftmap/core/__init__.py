# Copyright (C) 2024 Carnegie Mellon University

"""
SwiftMap Mapping Core Components

Core functionality for the VGGT mapping system including:
- Mapping session orchestrator (TCP transport + keyframe selection)
- TCP server for drone image reception
- Keyframe selection based on optical flow
- VGGT inference engine for 3D mapping
- Next Flight Navigation (NFN) planning
"""

from swiftmap.core.session import MappingSession
from swiftmap.core.tcp_server import MappingTCPServer
from swiftmap.core.keyframe_selector import KeyframeSelector
from swiftmap.core.mapper import (
    BaseMapper, VGGTMapper, VGGTOmegaMapper, get_mapper, available_mappers)
from swiftmap.core.nfn import NextFlightPlanner

__all__ = [
    "MappingSession",
    "MappingTCPServer",
    "KeyframeSelector",
    "BaseMapper",
    "VGGTMapper",
    "VGGTOmegaMapper",
    "get_mapper",
    "available_mappers",
    "NextFlightPlanner",
]