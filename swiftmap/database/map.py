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

    def __init__(self, name, path):
        self.meta = MapMeta(name=name, path=path, site=None, created_time=datetime.now())
        os.makedirs(self.images_dir, exist_ok=True)

        # optional
        self.reconstructed_results: PointCloud = None
        self.gps_aligned_results: Georeference = None
        self.next_flight_planned_results = None
        self.segmented_results = None

    @property
    def path(self) -> str:
        return self.meta.path

    @property
    def images_dir(self) -> str:
        return os.path.join(self.path, "images")

    def get_keyframe_paths(self) -> List[str]:
        """This map's keyframe JPEGs under images/, in capture order."""
        if not os.path.isdir(self.images_dir):
            return []
        return [os.path.join(self.images_dir, n) for n in sorted(os.listdir(self.images_dir))]

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
            pt.confidence_scene.export(os.path.join(self.path, "scene_confidence.glb"))

        if pt.camera_poses is not None:
            with open(os.path.join(self.path, "camera_poses.json"), "w") as f:
                json.dump(pt.camera_poses, f, indent=2)

        if pt.model_input is not None:
            model_input_dir = os.path.join(self.path, "model_input")
            os.makedirs(model_input_dir, exist_ok=True)
            for name, data in pt.model_input:
                with open(os.path.join(model_input_dir, name), "wb") as f:
                    f.write(data)