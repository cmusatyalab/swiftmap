# Copyright (C) 2024 Carnegie Mellon University
"""Next Flight Navigation (NFN).

Plans where to fly next: points inside the [P_low, P_high) confidence band are the
marginally-mapped regions, clustered into suggested viewpoints. See
``planner.NextFlightPlanner``."""

from .planner import NextFlightPlanner

__all__ = ["NextFlightPlanner"]
