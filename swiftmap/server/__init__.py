# Copyright (C) 2024 Carnegie Mellon University
"""Headless mapping server.

Collects frame+GPS pairs over TCP and, at the keyframe cap, runs the pipeline and grows
a single merged site. A Gradio page serves the results."""
from swiftmap.server.auto_server import AutoMappingServer, ServerConfig

__all__ = ["AutoMappingServer", "ServerConfig"]
