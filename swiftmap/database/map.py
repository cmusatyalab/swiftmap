# Copyright (C) 2024 Carnegie Mellon University

"""The ``Map``: one stored map directory in the database.

Holds a raw ``predictions.npz`` (a reconstructed batch) or a merged ``merged_points.npz``
plus transform/camera/map json. It is the store record -- identity, metadata, ``load``
(-> arrays) and ``write`` (arrays -> dir); previews and segmentation come from the pipeline."""


import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from swiftmap.database.types import GPS, Georeference, PointCloud


@dataclass
class MapMeta:
    name: str
    path: str
    site: Optional[str] = None
    created_time: datetime = field(default_factory=datetime.now)
    num_keyframes: int = 0
    gps_aligned: bool = False
    center_gps: Optional[GPS] = None
    geohash: Optional[str] = None

class Map:
    """An on-disk map directory (tag + path) with load/render/segment/merge I/O."""

    def __init__(self, name, path, keyframe_paths: Optional[List[str]] = None):
        self.meta = MapMeta(name=name, path=path, site=None, created_time=datetime.now(),
                            num_keyframes=len(keyframe_paths or []))
        self._keyframe_paths = keyframe_paths or []

        # optional
        self.reconstructed_results: PointCloud = None
        self.gps_aligned_results: Georeference = None
        self.next_flight_planned_results = None
        self.segmented_results = None

    @property
    def path(self) -> str:
        return self.meta.path

    def get_keyframe_paths(self) -> List[str]:
        return self._keyframe_paths

    def get_pointcloud(self) -> PointCloud:
        return self.reconstructed_results

    def get_georeference(self) -> Georeference:
        return self.gps_aligned_results

    def update_reconstruction(self, reconstruction: PointCloud):
        self.reconstructed_results = reconstruction

    def write2disk(self):
        """Export whatever's been built onto the pointcloud -- scene.glb,
        confidence_map.glb, camera_poses.json -- to disk under ``self.path``.

        The one place any of this map's results actually touch disk; postprocess's
        generate_*() functions only build and attach them to the ``PointCloud``.
        """
        pt = self.reconstructed_results
        if pt is None:
            return
        os.makedirs(self.path, exist_ok=True)

        if pt.scene is not None:
            pt.scene.export(os.path.join(self.path, "scene.glb"))

        if pt.confidence_scene is not None:
            pt.confidence_scene.export(os.path.join(self.path, "confidence_map.glb"))

        if pt.camera_poses is not None:
            with open(os.path.join(self.path, "camera_poses.json"), "w") as f:
                json.dump(pt.camera_poses, f, indent=2)