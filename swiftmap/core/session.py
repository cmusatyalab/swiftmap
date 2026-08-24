# Copyright (C) 2024 Carnegie Mellon University

"""The mapping session: the pipeline stages, composed.

It owns the map database and one instance of each pipeline stage -- reconstruction
backbone, GPS aligner, NFN planner, segmenter -- and runs them over a Map:

    process(map) = reconstruct() -> align_gps() -> plan()

The session holds no transport, threads or run state; ``swiftmap.server`` owns those
and hands finished Maps here. It is long-lived, so the (expensive) models survive
capture start/stop cycles.
"""

import os
from typing import Any, Dict, List, Optional

from swiftmap import constants
from swiftmap.database import Database
from swiftmap.database.map import Map
from swiftmap.core.pipeline.reconstructor import get_reconstructor, BaseReconstructor
from swiftmap.core.pipeline.next_flight_planner import NextFlightPlanner
# from swiftmap.core.pipeline.segmentor import get_segmenter, lift  # broken, not needed to start the session
from swiftmap.core.pipeline.gps_transformer import GpsTransformer


class MappingSession:
    """Owns the map database and one instance of each pipeline stage."""

    def __init__(self,
                 root: str = None,
                 site: str = "map",
                 backbone: List[str] = [constants.DEFAULT_RECONSTRUCTOR, constants.DEFAULT_SEGMENTER]):
        # db
        self.db = Database(root or os.getcwd(), site)

        # pipeline stages
        self.reconstructor: BaseReconstructor = get_reconstructor(backbone[0])
        self.aligner = GpsTransformer()
        self.planner = NextFlightPlanner()
        self.segmenter = None  # get_segmenter(backbone[1])  # broken, not needed to start the session

    # =============================================================== pipeline
    def process(self, map_: Map, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run every stage over one closed Map, stopping at the first failure."""
        result = self.reconstruct(map_, params or {})
        if not result.get("success"):
            return {"error": f"Reconstruction failed: {result.get('error')}"}

        aligned = self.align_gps(map_)
        if "error" in aligned:
            return {"error": f"GPS align failed: {aligned['error']}"}

        plan = self.plan(map_)
        if "error" in plan:
            return {"error": f"NFN failed: {plan['error']}"}

        map_.write2disk()
        return {"success": True, "map": map_}

    # ---------------------------------------------------------------- reconstruction
    def reconstruct(self, map_: Map, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run reconstruction on one closed Map."""
        print(f"Reconstructing {map_.meta.name} ({map_.meta.num_keyframes} keyframes)")
        return self.reconstructor.run(map_, params)

    # ------------------------------------------------------------ GPS alignment
    def align_gps(self, map_: Map, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fit the local->ENU transform for one Map from its keyframe GPS."""
        return self.aligner.run(map_, params)

    # ---------------------------------------------------------------- planning
    def plan(self, map_: Map, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plan the next flight for one Map (needs a Georeference)."""
        return self.planner.run(map_, params)

    # --------------------------------------------------------- segmentation
