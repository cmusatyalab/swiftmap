# Copyright (C) 2024 Carnegie Mellon University

"""
SwiftMap — AI-in-the-loop iterative drone mapping.

Turns a stream (or folder) of drone images + GPS into a dense 3D reconstruction,
a map-quality evaluation, and a next-flight plan:

1. Receive frame+GPS pairs over TCP.
2. Select keyframes by optical-flow disparity.
3. Reconstruct with a pluggable backbone (VGGT / VGGT-Omega).
4. Evaluate confidence, plan the next flight (NFN), and optionally segment
   objects by text query (SAM 3), with results exportable + GPS-tagged.

Two front ends share the same core: the Gradio GUI (``launch_mapping.py``) and
the headless auto-mapping server (``launch_server.py`` / ``swiftmap.server``).
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("swiftmap")
except PackageNotFoundError:      # running from source, not pip-installed
    __version__ = "0.0.0+dev"

# Main system components
from swiftmap.core.session import MappingSession
from swiftmap.core.transport.tcp_server import MappingTCPServer
from swiftmap.core.pipeline.keyframe_selector import KeyframeSelector
from swiftmap.core.pipeline.reconstructor import (
    BaseMapper, VGGTMapper, VGGTOmegaMapper, get_mapper, available_mappers)

# The Gradio GUI is optional: the headless mapping server (swiftmap.server) runs
# without gradio/viser installed, so importing the frontend must not be required.
try:
    from swiftmap.frontend.gradio_interface import MappingGradioInterface
except ImportError:
    MappingGradioInterface = None

# Keyframe selection helper
from swiftmap.core.pipeline.keyframe_selector import FrameTracker

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