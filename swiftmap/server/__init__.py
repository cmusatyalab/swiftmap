# Copyright (C) 2024 Carnegie Mellon University
"""Headless SwiftMap mapping server (no GUI).

Collects frame+GPS pairs over TCP (same wire protocol as the GUI), and when the
retained keyframe set reaches the cap, automatically runs the full pipeline —
reconstruction, GPS alignment, NFN, segmentation — and exports every artifact to
an output directory. Intended to run as a container that SteelEagle's SwiftMap
cognitive engine connects to (the way the SLAM engine connects to TerraSLAM).
"""
from swiftmap.server.auto_server import AutoMappingServer, ServerConfig

__all__ = ["AutoMappingServer", "ServerConfig"]
