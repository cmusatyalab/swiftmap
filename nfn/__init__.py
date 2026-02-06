"""
Next Flight Navigation (NFN) Module

Analyzes VGGT-generated 3D maps to guide drone flight planning by identifying
areas requiring additional coverage or higher quality mapping.
"""

from .gap_detector import CoverageGapDetector
from .quality_analyzer import QualityAnalyzer
from .viewpoint_planner import ViewpointPlanner

__all__ = [
    "CoverageGapDetector",
    "QualityAnalyzer",
    "ViewpointPlanner",
]
