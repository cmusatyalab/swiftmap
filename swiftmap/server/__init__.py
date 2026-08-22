# Copyright (C) 2024 Carnegie Mellon University
"""Headless mapping server.

Owns the TCP transport and keyframe selection, and drives each collected Map through
a ``MappingSession``. A Gradio page serves the results."""
from swiftmap.server.auto_server import AutoMappingServer, ServerConfig
from swiftmap.server.transport.transporter import Transporter
from swiftmap.server.transport.keyframe_selector import KeyframeSelector, FrameTracker

__all__ = ["AutoMappingServer", "ServerConfig", "Transporter", "KeyframeSelector", "FrameTracker"]
