# Copyright (C) 2024 Carnegie Mellon University

"""The ``Site``: a database's one growing map.

Every georeferenced ``Map`` is merged into a common ENU frame about the first map's
GPS origin, then collapsed onto a voxel grid so overlapping flights reinforce each
other instead of stacking duplicate points.
"""

import json
import os
from typing import List, Optional

import numpy as np
import trimesh

from swiftmap import constants
from swiftmap.database.map import Map
from swiftmap.database.types import GPS

_VOXEL_SIZE = 0.1       # metres; cells this size collapse to one point

# ENU (east, north, up) -> glTF's Y-up axes, matching the per-map georeferenced scene.
_ENU_TO_YUP = np.array([[-1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0]])


class Site:
    """Every aligned Map merged into one cloud, in ENU metres about a shared origin."""

    def __init__(self, path: str, voxel_size: float = _VOXEL_SIZE):
        self.path = path
        self.voxel_size = voxel_size
        self.maps: List[Map] = []
        self.origin: Optional[GPS] = None      # the first grown map's GPS origin
        self.points: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.conf: Optional[np.ndarray] = None
        self.source: Optional[np.ndarray] = None   # index into self.maps, per point

    def __repr__(self):
        return (f"Site(maps={len(self.maps)}, "
                f"points={0 if self.points is None else len(self.points)})")

    def grow(self, new_map: Map,
             conf_threshold: float = constants.DEFAULT_CONF_THRESHOLD) -> bool:
        """Merge one georeferenced Map in; False if it has nothing to contribute."""
        pt = new_map.get_pointcloud()
        georef = new_map.get_georeference()
        if pt is None or pt.world_points is None or georef is None:
            return False

        if self.origin is None:
            self.origin = georef.origin        # the first map fixes the common frame
        lla0 = (self.origin.latitude, self.origin.longitude, self.origin.altitude or 0.0)

        keep = pt.confidence_mask(conf_threshold)
        points = georef.to_enu(pt.flatten_points()[keep], lla0)
        colors = pt.flatten_colors()[keep]
        conf = pt.flatten_conf()[keep]
        source = np.full(len(points), len(self.maps), dtype=np.int64)

        if self.points is not None:
            points = np.vstack([self.points, points])
            colors = np.vstack([self.colors, colors])
            conf = np.concatenate([self.conf, conf])
            source = np.concatenate([self.source, source])

        self.points, self.colors, self.conf, self.source = _voxel_merge(
            points, colors, conf, source, self.voxel_size)
        self.maps.append(new_map)
        print(f"[site] grew with {new_map.meta.name}: {len(self.maps)} maps, "
              f"{len(self.points)} points")
        return True

    def write2disk(self):
        """Export the merged cloud twice -- true colour, and coloured by source map."""
        if self.points is None or not len(self.points):
            return
        os.makedirs(self.path, exist_ok=True)

        vertices = self.points @ _ENU_TO_YUP.T
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=self.colors),
                           geom_name="site")
        scene.export(os.path.join(self.path, "site.glb"))

        by_map = trimesh.Scene()
        by_map.add_geometry(trimesh.PointCloud(vertices=vertices, colors=self.map_colors()),
                            geom_name="merged")
        by_map.export(os.path.join(self.path, "merged_res.glb"))

        with open(os.path.join(self.path, "site.geojson"), "w") as f:
            json.dump(self.to_geojson(), f, indent=2)

    def map_colors(self) -> np.ndarray:
        """Per-point RGB naming which map each point came from, hues golden-angle apart."""
        import colorsys
        palette = np.array([[int(255 * v) for v in colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0,
                                                                      0.9, 1.0)]
                            for i in range(len(self.maps))], dtype=np.uint8)
        return palette[self.source]

    def to_geojson(self) -> dict:
        """site.geojson: where site.glb's local (0, 0, 0) sits on Earth."""
        return {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [self.origin.longitude, self.origin.latitude,
                                         self.origin.altitude or 0.0]},
            "properties": {"layer": "site_origin",
                           "model": "site.glb",
                           "description": "model local (0,0,0); the first merged map's origin",
                           "altitude_mode": "absolute",
                           "num_maps": len(self.maps),
                           "num_points": len(self.points)}}]}


def _voxel_merge(points, colors, conf, source, voxel_size: float):
    """Collapse points sharing a ``voxel_size`` grid cell into one, confidence-weighted.

    Position and color become the confidence-weighted mean of the cell; the merged
    confidence is the cell's max, and the cell is credited to the map that contributed
    its most confident point. ``voxel_size <= 0`` returns the input unchanged.
    """
    points = np.asarray(points, dtype=float)
    if voxel_size <= 0 or len(points) == 0:
        return points, np.asarray(colors), np.asarray(conf, dtype=float), np.asarray(source)

    w = np.asarray(conf, dtype=float)
    vidx = np.floor(points / voxel_size).astype(np.int64)
    vidx -= vidx.min(axis=0)
    dims = vidx.max(axis=0) + 1
    if dims.prod() >= np.iinfo(np.int64).max:
        raise OverflowError("voxel grid too large for a linear key; raise the voxel size")
    key = (vidx[:, 0] * dims[1] + vidx[:, 1]) * dims[2] + vidx[:, 2]
    _, inv = np.unique(key, return_inverse=True)

    g = inv.max() + 1
    wsum = np.bincount(inv, weights=w, minlength=g)
    wsafe = np.where(wsum > 0, wsum, 1.0)
    pos = np.empty((g, 3))
    col = np.empty((g, 3))
    cols_f = np.asarray(colors, dtype=float)
    for k in range(3):
        pos[:, k] = np.bincount(inv, weights=w * points[:, k], minlength=g) / wsafe
        col[:, k] = np.bincount(inv, weights=w * cols_f[:, k], minlength=g) / wsafe
    mconf = np.zeros(g)
    np.maximum.at(mconf, inv, w)
    # visiting cells in ascending confidence leaves each one holding its best point's map
    order = np.argsort(w, kind="stable")
    msrc = np.zeros(g, dtype=np.int64)
    msrc[inv[order]] = np.asarray(source)[order]
    return pos, np.clip(col, 0, 255).astype(np.uint8), mconf, msrc
