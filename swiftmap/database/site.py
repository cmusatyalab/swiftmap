# Copyright (C) 2024 Carnegie Mellon University

"""The ``Site``: a database's one growing map.

Every georeferenced ``Map`` is merged into a common ENU frame about the first map's
GPS origin, then collapsed onto a voxel grid so overlapping flights reinforce each
other instead of stacking duplicate points.
"""

import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np
import trimesh

from swiftmap import constants
from swiftmap.database.map import Map
from swiftmap.database.types import GPS

_VOXEL_SIZE = 0.1       # metres; cells this size collapse to one point
# Confidence colours span a fixed range, not each write's own min/max, so the ramp means
# the same thing every iteration and the site can be compared with itself as it grows.
_CONF_RANGE = (0.0, 25.0)

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

    def drop(self, dropped: Map) -> bool:
        """Remove one map's contribution and re-index the rest"""
        if dropped not in self.maps:
            return False

        index = self.maps.index(dropped)
        keep = self.source != index
        self.points, self.colors = self.points[keep], self.colors[keep]
        self.conf, self.source = self.conf[keep], self.source[keep]
        self.source[self.source > index] -= 1      # close the gap the removal leaves
        self.maps.pop(index)
        print(f"[site] dropped {dropped.meta.name}: {len(self.maps)} maps, "
              f"{len(self.points)} points")
        return True

    def write2disk(self):
        """Export the merged cloud twice -- true colour, and coloured by source map."""
        if self.points is None or not len(self.points):
            shutil.rmtree(self.path, ignore_errors=True)   # nothing left; drop stale files
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

        by_conf = trimesh.Scene()
        by_conf.add_geometry(trimesh.PointCloud(vertices=vertices, colors=self.conf_colors()),
                             geom_name="confidence")
        by_conf.export(os.path.join(self.path, "site_confidence.glb"))

        with open(os.path.join(self.path, "site.geojson"), "w") as f:
            json.dump(self.to_geojson(), f, indent=2)

        # the GLBs carry only position and colour; keep what a future merge needs
        np.savez_compressed(os.path.join(self.path, "site_points.npz"),
                            conf=self.conf.astype(np.float32), source=self.source)

    def conf_colors(self) -> np.ndarray:
        """Per-point RGBA over a fixed confidence range: red (low) to green (high)."""
        lo, hi = _CONF_RANGE
        norm = np.clip((self.conf - lo) / (hi - lo), 0.0, 1.0)
        colors = np.zeros((len(norm), 4), dtype=np.uint8)
        colors[:, 0] = ((1.0 - norm) * 255).astype(np.uint8)
        colors[:, 1] = (norm * 255).astype(np.uint8)
        colors[:, 3] = 255
        return colors

    def map_colors(self) -> np.ndarray:
        """Per-point RGB naming which map each point came from."""
        return self._palette()[self.source]

    def _palette(self) -> np.ndarray:
        """One RGB per merged map, hues golden-angle apart so any count stays distinct."""
        return np.array([self._palette_at(i) for i in range(max(len(self.maps), 1))],
                        dtype=np.uint8)

    @staticmethod
    def _palette_at(index: int) -> np.ndarray:
        """The index-th distinct hue, spaced by the golden angle."""
        import colorsys
        return np.array([int(255 * v) for v in
                         colorsys.hsv_to_rgb((index * 0.6180339887) % 1.0, 0.9, 1.0)],
                        dtype=np.uint8)

    def load(self, maps: Dict[str, Map]) -> bool:
        """Restore a site an earlier run wrote; False if there is nothing on disk."""
        meta = os.path.join(self.path, "site.geojson")
        cloud = os.path.join(self.path, "site.glb")
        if not all(os.path.isfile(f) for f in
                   (meta, cloud, os.path.join(self.path, "site_points.npz"))):
            return False

        with open(meta) as f:
            feature = json.load(f)["features"][0]
        lon, lat, alt = feature["geometry"]["coordinates"]
        self.origin = GPS(lat, lon, alt)
        self.maps = [maps[i] for i in feature["properties"].get("map_ids", []) if i in maps]

        geometry = list(trimesh.load(cloud).geometry.values())[0]
        self.points = np.asarray(geometry.vertices) @ np.linalg.inv(_ENU_TO_YUP).T
        self.colors = np.asarray(geometry.colors)[:, :3]

        arrays = np.load(os.path.join(self.path, "site_points.npz"))
        self.conf = arrays["conf"].astype(float)
        self.source = arrays["source"]
        print(f"[site] loaded {len(self.points)} points from {len(self.maps)} maps")
        return True

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
                           "num_points": len(self.points),
                           "mean_confidence": float(self.conf.mean()),
                           "median_confidence": float(np.median(self.conf)),
                           "map_ids": [m.meta.name for m in self.maps]}}]}


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
