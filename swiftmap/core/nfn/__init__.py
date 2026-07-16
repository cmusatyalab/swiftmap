# Copyright (C) 2024 Carnegie Mellon University
"""
Next Flight Navigation (NFN)

Plans where a drone should fly next to improve a VGGT reconstruction, using the
confidence-difference of the map: points in the [P_low, P_high) confidence band are
clustered and turned into suggested viewpoints. See planner.NextFlightPlanner.
"""

from .planner import NextFlightPlanner

__all__ = ["NextFlightPlanner"]
