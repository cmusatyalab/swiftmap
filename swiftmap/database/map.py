# Copyright (C) 2024 Carnegie Mellon University

"""The ``Map``: one stored map directory in the database.

Holds the batch's keyframes (``images/``) with their paired GPS, plus whatever the
pipeline stages attach: a ``PointCloud`` reconstruction, a ``Georeference``, and a
``FlightPlan``. ``write2disk()`` is the one place any of it touches disk."""


import csv
import json
import os
import re

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from swiftmap.database.types import (CameraPose, GPS, FlightPlan, Georeference,
                                     PointCloud)


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

        # input
        self.keyframe_images: List[str] = []
        self.keyframe_gps: List[Optional[GPS]] = []

        # processed
        self.reconstructed_results: PointCloud = None
        self.gps_aligned_results: Georeference = None
        self.flight_plan: FlightPlan = None
        self.segmented_results = None

    @property
    def path(self) -> str:
        return self.meta.path

    @property
    def input_dir(self) -> str:
        return os.path.join(self.path, "raw")

    @property
    def pointcloud_dir(self) -> str:
        return os.path.join(self.path, "pointcloud")
    
    @property
    def georeference_dir(self) -> str:
        return os.path.join(self.path, "georeference")

    @property
    def flightplan_dir(self) -> str:
        return os.path.join(self.path, "nextflightplan")
    
    @property
    def images_dir(self) -> str:
        return os.path.join(self.input_dir, "images")

    @property
    def gps_dir(self) -> str:
        return os.path.join(self.input_dir, "gps")

    @property
    def local_dir(self) -> str:
        """Scenes in the reconstruction's own unitless, camera-aligned coordinates."""
        return os.path.join(self.pointcloud_dir, "local")

    @property
    def georeferenced_dir(self) -> str:
        """The same surface in ENU metres, placed on Earth."""
        return os.path.join(self.pointcloud_dir, "georeferenced")

    @property
    def inference_dir(self) -> str:
        """What the backbone saw and predicted: the frames, and the per-pixel grid."""
        return os.path.join(self.pointcloud_dir, "inference")

    @property
    def model_input_dir(self) -> str:
        return os.path.join(self.inference_dir, "model_input")

# ================================================================== input
    def get_input_keyframe_paths(self) -> List[str]:
        """This map's keyframe JPEGs under images/, in capture order."""
        if not os.path.isdir(self.images_dir):
            return []
        return [os.path.join(self.images_dir, n) for n in sorted(os.listdir(self.images_dir))]

    def get_input_gps_path(self) -> str:
        return os.path.join(self.gps_dir, "input_gps.csv")

    def get_input_gps(self) -> List[Optional[GPS]]:
        return self.keyframe_gps

    def update_input(self, images: List[str], gps: List[Optional[GPS]]):
        """Record the batch's keyframe filenames and their paired GPS (1:1, capture order)."""
        self.keyframe_images = list(images)
        self.keyframe_gps = list(gps)
        self.meta.num_keyframes = len(self.keyframe_images)

# ================================================================== point cloud
    def get_pointcloud(self) -> PointCloud:
        return self.reconstructed_results


    def update_pointcloud(self, reconstruction: PointCloud):
        self.reconstructed_results = reconstruction

# ================================================================== plan        
    def get_flight_plan(self) -> FlightPlan:
        return self.flight_plan

    def update_flight_plan(self, flight_plan: FlightPlan):
        self.flight_plan = flight_plan

# ================================================================== geo
    def get_georeference(self) -> Georeference:
        return self.gps_aligned_results

    def update_georeference(self, georeference: Georeference):
        self.gps_aligned_results = georeference
        self.meta.gps_aligned = georeference is not None



# ================================================================== load

    def load(self) -> "Map":
        """Restore what an earlier run left under ``self.path``."""
        images = [os.path.basename(p) for p in self.get_input_keyframe_paths()]
        self.keyframe_images = list(images)
        self.keyframe_gps = self._read_input_gps(len(self.keyframe_images))
        self.meta.num_keyframes = len(self.keyframe_images)
        self._load_pointcloud()
        self._load_georeference()
        self._load_flight_plan()
        return self
    
    def _read_input_gps(self, num_keyframes: int) -> List[Optional[GPS]]:
        """The GPS write2disk() stored, one per keyframe in capture order."""
        path = self.get_input_gps_path()
        if not os.path.isfile(path):
            return [None] * num_keyframes

        rows: List[Optional[GPS]] = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)                                  # header
            for row in reader:
                if len(row) < 2 or not row[0] or not row[1]:
                    rows.append(None)                           # keyframe with no fix
                    continue
                alt = row[2] if len(row) > 2 else ""
                rows.append(GPS(float(row[0]), float(row[1]),
                                float(alt) if alt else None))

        if len(rows) != num_keyframes:
            print(f"{path}: {len(rows)} gps rows for {num_keyframes} keyframes, ignoring")
            return [None] * num_keyframes
        return rows
    
    def _load_flight_plan(self):
        """Rebuild the plan from nextflightplan/next_flight.kml"""
        path = os.path.join(self.flightplan_dir, "next_flight.kml")
        if not os.path.isfile(path):
            return
        with open(path) as f:
            fixes = re.findall(
                r"<Point>.*?<coordinates>([-\d.]+),([-\d.]+),([-\d.]+)</coordinates>", f.read())
        if fixes:
            self.update_flight_plan(FlightPlan(waypoints=[
                GPS(float(lat), float(lon), float(alt)) for lon, lat, alt in fixes]))

    def _load_pointcloud(self):
        """Rebuild the reconstruction from inference/prediction.npz"""
        path = os.path.join(self.inference_dir, "prediction.npz")
        if not os.path.isfile(path):
            return
        data = np.load(path)
        held = {name: data[name] for name in data.files}
        extrinsic = held.get("extrinsic")
        self.reconstructed_results = PointCloud(
            **held,
            cameras=[] if extrinsic is None else
            [CameraPose.from_extrinsic(e) for e in extrinsic])

    def _load_georeference(self):
        """Rebuild the transform from georeference/transform.json"""
        path = os.path.join(self.georeference_dir, "transform.json")
        if not os.path.isfile(path):
            return
        with open(path) as f:
            cfg = json.load(f)
        origin = cfg["origin"]
        self.update_georeference(Georeference(
            cfg["scale"], cfg["rotation"], cfg["translation"],
            GPS(origin["latitude"], origin["longitude"], origin["altitude"])))


# ================================================================== save
    def write2disk(self):
        """Export everything the pipeline stages attached, under ``self.path``."""
        os.makedirs(self.path, exist_ok=True)

        if self.keyframe_gps:
            os.makedirs(self.gps_dir, exist_ok=True)
            with open(self.get_input_gps_path(), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["latitude", "longitude", "altitude"])
                for gps in self.keyframe_gps:
                    if gps is None:
                        writer.writerow(["", "", ""])
                    else:
                        writer.writerow([gps.latitude, gps.longitude,
                                         "" if gps.altitude is None else gps.altitude])

        pt = self.reconstructed_results
        if pt is not None:
            if pt.scene is not None:
                pt.scene.export(self._artifact(self.local_dir, "scene.glb"))

            if pt.confidence_scene is not None:
                pt.confidence_scene.export(
                    self._artifact(self.local_dir, "scene_confidence.glb"))

            if pt.segmented_scene is not None:
                pt.segmented_scene.export(self._artifact(self.local_dir, "seg_scene.glb"))

            if pt.model_input is not None:
                for name, data in pt.model_input:
                    with open(self._artifact(self.model_input_dir, name), "wb") as f:
                        f.write(data)

            prediction = {name: array for name, array in (
                ("world_points", pt.world_points),
                ("world_points_conf", pt.world_points_conf),
                ("images", pt.images),
                ("extrinsic", pt.extrinsic),
                ("intrinsic", pt.intrinsic)) if array is not None}
            if prediction:
                np.savez_compressed(self._artifact(self.inference_dir, "prediction.npz"),
                                    **prediction)

            if pt.geo_scene is not None:
                pt.geo_scene.export(self._artifact(self.georeferenced_dir, "scene_geo.glb"))
                with open(self._artifact(self.georeferenced_dir, "scene_geo.geojson"), "w") as f:
                    json.dump(pt.to_geojson(self.gps_aligned_results), f, indent=2)
                cam_kml = pt.cameras_to_kml(self.gps_aligned_results)
                if cam_kml is not None:
                    with open(self._artifact(self.georeferenced_dir, "camera_poses.kml"), "w") as f:
                        f.write(cam_kml)

        if self.gps_aligned_results is not None:
            with open(self._artifact(self.georeference_dir, "transform.json"), "w") as f:
                json.dump(self.gps_aligned_results.to_json(), f, indent=2)

        if self.flight_plan is not None:
            kml = self.flight_plan.to_kml()
            if kml is not None:
                with open(self._artifact(self.flightplan_dir, "next_flight.kml"), "w") as f:
                    f.write(kml)

    @staticmethod
    def _artifact(directory: str, name: str) -> str:
        """Path to an artifact, creating the directory it belongs in."""
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, name)
