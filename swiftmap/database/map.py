# Copyright (C) 2024 Carnegie Mellon University

"""The ``Map``: one stored map directory in the database.

Holds the batch's keyframes (``images/``) with their paired GPS, plus whatever the
pipeline stages attach: a ``PointCloud`` reconstruction, a ``Georeference``, and a
``FlightPlan``. ``write2disk()`` is the one place any of it touches disk."""


import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from swiftmap.database.types import GPS, FlightPlan, Georeference, PointCloud


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

        # keyframes: the two lists are 1:1, in capture order
        self.keyframe_images: List[str] = []
        self.keyframe_gps: List[Optional[GPS]] = []

        # optional
        self.reconstructed_results: PointCloud = None
        self.gps_aligned_results: Georeference = None
        self.flight_plan: FlightPlan = None
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

    def get_keyframe_gps(self) -> List[Optional[GPS]]:
        return self.keyframe_gps

    def update_keyframes(self, images: List[str], gps: List[Optional[GPS]]):
        """Record the batch's keyframe filenames and their paired GPS (1:1, capture order)."""
        self.keyframe_images = list(images)
        self.keyframe_gps = list(gps)
        self.meta.num_keyframes = len(self.keyframe_images)

    def get_pointcloud(self) -> PointCloud:
        return self.reconstructed_results

    def get_flight_plan(self) -> FlightPlan:
        return self.flight_plan

    def get_georeference(self) -> Georeference:
        return self.gps_aligned_results

    def update_reconstruction(self, reconstruction: PointCloud):
        self.reconstructed_results = reconstruction

    def update_georeference(self, georeference: Georeference):
        self.gps_aligned_results = georeference
        self.meta.gps_aligned = georeference is not None

    def update_flight_plan(self, flight_plan: FlightPlan):
        self.flight_plan = flight_plan

    def write2disk(self):
        """Export everything the pipeline stages attached, under ``self.path``."""
        os.makedirs(self.path, exist_ok=True)

        pt = self.reconstructed_results
        if pt is not None:
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

        if self.gps_aligned_results is not None:
            with open(os.path.join(self.path, "transform.json"), "w") as f:
                json.dump(self.gps_aligned_results.to_json(), f, indent=2)

        if self.flight_plan is not None:
            with open(os.path.join(self.path, "next_flight_viewpoints.json"), "w") as f:
                json.dump(self.flight_plan.to_json(), f, indent=2)
            kml = self.flight_plan.to_kml()
            if kml is not None:
                with open(os.path.join(self.path, "next_flight_viewpoints.kml"), "w") as f:
                    f.write(kml)
