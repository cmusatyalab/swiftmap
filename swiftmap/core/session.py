# Copyright (C) 2024 Carnegie Mellon University

"""The mapping session: the pipeline stages, composed.

It holds the server's database and one instance of each pipeline stage -- reconstruction
backbone, GPS aligner, NFN planner, segmenter -- and runs them over a Map. Stages attach
their results to the Map; storing it is the server's job.

    process(map_id) = reconstruct() -> align_gps() -> plan()

Maps are passed between the transporter, server and session by id; the session
resolves them through the database.

The session holds no transport, threads or run state; ``swiftmap.server`` owns those
and hands finished Maps here. It is long-lived, so the (expensive) models survive
capture start/stop cycles.
"""

from typing import Any, Dict, List, Optional

from swiftmap import constants
from swiftmap.database import Database
from swiftmap.database.map import Map
from swiftmap.core.pipeline.reconstructor import get_reconstructor, BaseReconstructor
from swiftmap.core.pipeline.next_flight_planner import NextFlightPlanner
from swiftmap.core.pipeline.segmentor import get_segmenter, BaseSegmenter
from swiftmap.core.pipeline.gps_transformer import GpsTransformer


class MappingSession:
    """Holds the server's database and one instance of each pipeline stage."""

    def __init__(self,
                 db: Database,
                 backbone: List[str] = [constants.DEFAULT_RECONSTRUCTOR, constants.DEFAULT_SEGMENTER]):
        self.db = db          # the server's database, shared with the transporter

        # pipeline stages
        self.reconstructor: BaseReconstructor = get_reconstructor(backbone[0])
        self.aligner = GpsTransformer()
        self.planner = NextFlightPlanner()
        self.segmenter: BaseSegmenter = get_segmenter(backbone[1])

    # =============================================================== pipeline
    def process(self, map_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run every stage over one closed Map, stopping at the first failure."""
        map_ = self.db.get_map(map_id)
        if map_ is None:
            return {"error": f"Unknown map '{map_id}'"}

        result = self.reconstruct(map_, params or {})
        if not result.get("success"):
            return {"error": f"Reconstruction failed: {result.get('error')}"}

        aligned = self.align_gps(map_)
        if "error" in aligned:
            return {"error": f"GPS align failed: {aligned['error']}"}

        plan = self.plan(map_)
        if "error" in plan:
            return {"error": f"NFN failed: {plan['error']}"}
        return {"success": True, "map_id": map_id}

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
    def segment(self, map_: Map, query: str,
                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Segment one text query over a reconstructed Map and lift it into 3D."""
        return self.segmenter.run(map_, {**(params or {}), "query": query})
