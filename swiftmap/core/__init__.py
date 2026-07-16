# Copyright (C) 2024 Carnegie Mellon University

"""
SwiftMap core components:

- Mapping session orchestrator (TCP transport + keyframe selection + backbones)
- TCP server for drone frame+GPS reception
- Optical-flow keyframe selection
- Pluggable reconstruction backbones (VGGT / VGGT-Omega)
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