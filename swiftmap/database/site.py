# Copyright (C) 2024 Carnegie Mellon University

"""The ``Site``: a database's one growing map.

Every georeferenced ``Map`` is merged into a common ENU frame about the first map's
GPS origin, then collapsed onto a voxel grid where each cell keeps its most confident
observation, so overlapping flights do not stack duplicate points. The site stays
sorted by cell, so a merge sorts only the incoming batch, not the whole site.
"""

import json
import os
import shutil
from typing import Dict, List, Optional

import numpy as np
import trimesh

from swiftmap import constants
from swiftmap.database.map import Map
from swiftmap.database.types import GPS, SegmentedTarget

_VOXEL_SIZE = 0.1       # metres; cells this size collapse to one point
_KEY_OFFSET = 1 << 20   # cell keys pack 21 bits per axis: +-2^20 cells (+-104 km at 0.1 m)
_BACKGROUND_DIM = 0.35  # segmented targets should read against a muted site
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
        self.keys: Optional[np.ndarray] = None     # cell key per point, ascending
        self.points: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.conf: Optional[np.ndarray] = None
        self.source: Optional[np.ndarray] = None   # index into self.maps, per point
        self.segments: List[SegmentedTarget] = []  # one per text query, kept across queries

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
        self._merge(points, pt.flatten_colors()[keep], pt.flatten_conf()[keep],
                    np.full(len(points), len(self.maps), dtype=np.int64))
        self.maps.append(new_map)
        print(f"[site] grew with {new_map.meta.name}: {len(self.maps)} maps, "
              f"{len(self.points)} points")
        return True

    def _merge(self, points, colors, conf, source):
        """Fold a batch in, keeping each cell's most confident point.

        Only the batch is sorted. Its cells are found in the site by binary search,
        replaced where the batch is more confident, and inserted where new. A maximum
        does not depend on grouping, so this gives exactly the site that merging every
        batch at once would; on a tie the existing site point stays.
        """
        keys = _cell_keys(points, self.voxel_size)
        best = _best_per_cell(keys, conf)
        batch = {"keys": keys[best], "points": points[best], "colors": colors[best],
                 "conf": conf[best], "source": source[best]}
        if self.keys is None:
            for name, values in batch.items():
                setattr(self, name, values)
            return

        pos = np.searchsorted(self.keys, batch["keys"])
        hit = pos < len(self.keys)
        hit[hit] = self.keys[pos[hit]] == batch["keys"][hit]
        better = np.zeros_like(hit)
        better[hit] = batch["conf"][hit] > self.conf[pos[hit]]
        new = ~hit
        for name, values in batch.items():
            current = getattr(self, name)
            current[pos[better]] = values[better]
            setattr(self, name, np.insert(current, pos[new], values[new], axis=0))

    def drop(self, dropped: Map) -> bool:
        """Remove one map's contribution and re-index the rest"""
        if dropped not in self.maps:
            return False

        index = self.maps.index(dropped)
        keep = self.source != index                # a filter keeps the cell order
        self.keys, self.points = self.keys[keep], self.points[keep]
        self.colors, self.conf, self.source = self.colors[keep], self.conf[keep], self.source[keep]
        self.source[self.source > index] -= 1      # close the gap the removal leaves
        self.maps.pop(index)
        self.segments = []      # segments carry no source, so the dropped map's would linger
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

        seg_glb = os.path.join(self.path, "seg_site.glb")
        if not self.segments and os.path.isfile(seg_glb):
            os.remove(seg_glb)          # nothing left to show; drop the stale scene
        if self.segments:
            by_seg = trimesh.Scene()
            background = (self.colors * _BACKGROUND_DIM).astype(np.uint8)
            by_seg.add_geometry(trimesh.PointCloud(vertices=vertices, colors=background),
                                geom_name="site")
            for target in self.segments:
                if not len(target.points):
                    continue
                by_seg.add_geometry(
                    trimesh.PointCloud(vertices=target.points @ _ENU_TO_YUP.T,
                                       colors=np.tile(target.color, (len(target.points), 1))),
                    geom_name=target.query)      # node name is the viewer's label
            by_seg.export(seg_glb)

        np.savez_compressed(os.path.join(self.path, "site_points.npz"),
                            conf=self.conf.astype(np.float32), source=self.source,
                            keys=self.keys)
        np.savez_compressed(os.path.join(self.path, "site_segments.npz"),
                            queries=np.array([t.query for t in self.segments], dtype=object),
                            colors=np.array([t.color for t in self.segments]).reshape(-1, 3),
                            counts=np.array([len(t.points) for t in self.segments]),
                            points=(np.vstack([t.points for t in self.segments])
                                    if self.segments else np.zeros((0, 3))))

    def add_segmentation(self, query: str, points: np.ndarray) -> SegmentedTarget:
        """Store one query's lifted points, giving it a hue no earlier target has used.

        Targets accumulate across queries; re-asking an earlier query replaces it.
        """
        import colorsys
        self.segments = [t for t in self.segments if t.query != query]
        hue = (len(self.segments) * 0.6180339887) % 1.0      # golden-angle spacing
        color = np.array([int(255 * v) for v in colorsys.hsv_to_rgb(hue, 0.9, 1.0)],
                         dtype=np.uint8)
        target = SegmentedTarget(query, points, color)
        self.segments.append(target)
        return target

    def clear_segmentations(self) -> int:
        """Drop every segmented target; how many were removed."""
        count = len(self.segments)
        self.segments = []
        return count

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
        arrays = np.load(os.path.join(self.path, "site_points.npz"))
        # the keys are stored, not re-derived: the GLB holds float32, and rounding can
        # carry a point across a cell boundary
        self.keys = arrays["keys"]
        self.points = np.asarray(geometry.vertices) @ np.linalg.inv(_ENU_TO_YUP).T
        self.colors = np.array(geometry.colors)[:, :3]
        self.conf = arrays["conf"].astype(float)
        self.source = arrays["source"]
        self._load_segments()
        print(f"[site] loaded {len(self.points)} points from {len(self.maps)} maps, "
              f"{len(self.segments)} segmented target(s)")
        return True

    def _load_segments(self):
        """Restore the accumulated segmentation targets, if an earlier query wrote any."""
        path = os.path.join(self.path, "site_segments.npz")
        if not os.path.isfile(path):
            return
        arrays = np.load(path, allow_pickle=True)
        bounds = np.cumsum(arrays["counts"])[:-1]
        self.segments = [SegmentedTarget(str(query), points, color)
                         for query, color, points in
                         zip(arrays["queries"], arrays["colors"],
                             np.split(arrays["points"], bounds))]

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


def _cell_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """One int64 per point naming its ``voxel_size`` grid cell.

    The packing is fixed, not fitted to the input's extent, so keys from different
    merges compare directly.
    """
    cells = np.floor(np.asarray(points, dtype=float) / voxel_size).astype(np.int64) + _KEY_OFFSET
    if len(cells) and (cells.min() < 0 or cells.max() >= 2 * _KEY_OFFSET):
        raise OverflowError("site wider than the cell keys cover; raise the voxel size")
    return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]


def _best_per_cell(keys: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Index of each cell's most confident point, in ascending cell order.

    The sort is stable, so on a tie the earlier point wins.
    """
    order = np.lexsort((-np.asarray(conf, dtype=float), keys))
    first = np.ones(len(order), dtype=bool)
    first[1:] = keys[order[1:]] != keys[order[:-1]]
    return order[first]
