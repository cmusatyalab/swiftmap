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
from swiftmap.database.types import GPS, SegmentedInstance, SegmentedTarget

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

_FIELDS = ("keys", "points", "colors", "conf", "source")
_VIEWS = ("scene", "confidence", "merged")
_LABEL_HEIGHT = 2.0
_DIGITS = {"0": ("111", "101", "101", "101", "111"), "1": ("010", "110", "010", "010", "111"),
           "2": ("111", "001", "111", "100", "111"), "3": ("111", "001", "111", "001", "111"),
           "4": ("101", "101", "111", "001", "001"), "5": ("111", "100", "111", "001", "111"),
           "6": ("111", "100", "111", "101", "111"), "7": ("111", "001", "001", "001", "001"),
           "8": ("111", "101", "111", "101", "111"), "9": ("111", "101", "111", "001", "111")}


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

    # -------------------------------- helper --------------------------------
    def _stage_dir(self, upto: int) -> str:
        """Where the site after its first ``upto`` maps is stored."""
        return os.path.join(self.path, "stages", str(upto))

    # -------------------------------- grow and shrink --------------------------------
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
        keys = self._cell_keys(points, self.voxel_size)
        best = self._best_per_cell(keys, conf)
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

    def _cell_keys(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """One 64-bit integer per point, encoding the cell it falls into."""
        cells = np.floor(np.asarray(points, dtype=float) / voxel_size).astype(np.int64) + _KEY_OFFSET
        if len(cells) and (cells.min() < 0 or cells.max() >= 2 * _KEY_OFFSET):
            raise OverflowError("site wider than the cell keys cover; raise the voxel size")
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

    def _best_per_cell(self, keys: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """Return the index of each cell's most confident point, in ascending cell order."""
        order = np.lexsort((-np.asarray(conf, dtype=float), keys))
        first = np.ones(len(order), dtype=bool)
        first[1:] = keys[order[1:]] != keys[order[:-1]]
        return order[first]

    def drop(self, dropped: Map) -> bool:
        """Remove one map's points, re-index the rest, and delete the stages that held it."""
        if dropped not in self.maps:
            return False

        index = self.maps.index(dropped)
        keep = self.source != index
        for name in _FIELDS:
            setattr(self, name, getattr(self, name)[keep])
        self.source[self.source > index] -= 1
        for stage in range(index + 1, len(self.maps) + 1):
            shutil.rmtree(self._stage_dir(stage), ignore_errors=True)
        self.maps.pop(index)
        self.segments = []      # segments carry no source, so the dropped map's would linger
        print(f"[site] dropped {dropped.meta.name}: {len(self.maps)} maps, "
              f"{len(self.points)} points")
        return True

    # -------------------------------- stages --------------------------------
    def stage_scene(self, upto: int, view: str) -> Optional[str]:
        """Path to the ``view`` GLB of the site after its first ``upto`` maps."""
        path = os.path.join(self._stage_dir(upto), f"{view}.glb")
        return path if os.path.isfile(path) else None

    # -------------------------------- segmentation --------------------------------
    def add_segmentation(self, query: str,
                         instances: List[SegmentedInstance]) -> SegmentedTarget:
        """Store one query's instances under a hue no earlier target used; a repeat replaces it."""
        import colorsys
        self.segments = [t for t in self.segments if t.query != query]
        hue = (len(self.segments) * 0.6180339887) % 1.0      # golden-angle spacing
        color = np.array([int(255 * v) for v in colorsys.hsv_to_rgb(hue, 0.9, 1.0)],
                         dtype=np.uint8)
        target = SegmentedTarget(query, color, instances)
        self.segments.append(target)
        return target

    def clear_segmentations(self) -> int:
        """Drop every segmented target; how many were removed."""
        count = len(self.segments)
        self.segments = []
        return count

    # -------------------------------- persistence --------------------------------
    def write2disk(self):
        """Save the site as its latest stage, with the files every stage shares."""
        if self.points is None or not len(self.points):
            shutil.rmtree(self.path, ignore_errors=True)   # nothing left; drop stale files
            return
        stage = self._stage_dir(len(self.maps))
        os.makedirs(stage, exist_ok=True)

        fields = {name: getattr(self, name) for name in _FIELDS}
        for view in _VIEWS:
            self._export_cloud(os.path.join(stage, f"{view}.glb"), self.points,
                               self._view_colors(view, fields), view)
        np.savez_compressed(os.path.join(stage, "site.npz"), **fields)

        with open(os.path.join(self.path, "site.geojson"), "w") as f:
            json.dump(self.to_geojson(), f, indent=2)

        seg_glb = os.path.join(self.path, "seg_site.glb")
        if not self.segments and os.path.isfile(seg_glb):
            os.remove(seg_glb)          # nothing left to show; drop the stale scene
        if self.segments:
            by_seg = trimesh.Scene()
            background = (self.colors * _BACKGROUND_DIM).astype(np.uint8)
            by_seg.add_geometry(trimesh.PointCloud(vertices=self.points @ _ENU_TO_YUP.T,
                                                   colors=background),
                                geom_name="site")
            for target in self.segments:
                for n, inst in enumerate(target.instances):
                    shade = (target.color * (1.0 - 0.25 * (n % 3))).astype(np.uint8)
                    by_seg.add_geometry(
                        trimesh.PointCloud(vertices=inst.points @ _ENU_TO_YUP.T,
                                           colors=np.tile(shade, (len(inst.points), 1))),
                        geom_name=f"{target.query}/{n}")
                    above = np.append(inst.centroid[:2], inst.points[:, 2].max() + _LABEL_HEIGHT)
                    label = self._label_points(str(n), above)
                    by_seg.add_geometry(
                        trimesh.PointCloud(vertices=label @ _ENU_TO_YUP.T,
                                           colors=np.tile(shade, (len(label), 1))),
                        geom_name=f"{target.query}/{n}/label")
            by_seg.export(seg_glb)

        owned = [(t, inst) for t, target in enumerate(self.segments) for inst in target.instances]
        np.savez_compressed(os.path.join(self.path, "site_segments.npz"),
                            queries=np.array([t.query for t in self.segments], dtype=object),
                            colors=np.array([t.color for t in self.segments]).reshape(-1, 3),
                            owner=np.array([t for t, _ in owned], dtype=np.int64),
                            support=np.array([i.support for _, i in owned], dtype=np.int64),
                            score=np.array([i.score for _, i in owned], dtype=float),
                            counts=np.array([len(i.points) for _, i in owned], dtype=np.int64),
                            points=(np.vstack([i.points for _, i in owned])
                                    if owned else np.zeros((0, 3))))

    @staticmethod
    def _label_points(text: str, at: np.ndarray) -> np.ndarray:
        """``text`` as solid dots in a 3x5 font, lying flat and centred on ``at``, north up."""
        cell = _LABEL_HEIGHT / 5
        fill = np.stack(np.meshgrid(np.linspace(0, 0.9, 6), np.linspace(0, 0.9, 6)), -1).reshape(-1, 2)
        pixels = np.array([(4 * c + x, 4 - y) for c, char in enumerate(text)
                           for y, row in enumerate(_DIGITS[char])
                           for x, bit in enumerate(row) if bit == "1"], dtype=float)
        xy = (pixels[:, None, :] + fill[None]).reshape(-1, 2) * cell
        xy -= [(4 * len(text) - 1) * cell / 2, 2.5 * cell]
        return np.column_stack([xy + at[:2], np.full(len(xy), at[2])])

    def _view_colors(self, view: str, fields: Dict[str, np.ndarray]) -> np.ndarray:
        """Per-point colours of ``view``: true colour, confidence, or source map."""
        if view == "confidence":
            return self.conf_colors(fields["conf"])
        if view == "merged":
            return self.map_colors(fields["source"])
        return fields["colors"]

    def conf_colors(self, conf: np.ndarray) -> np.ndarray:
        """Per-point RGBA over a fixed confidence range: red (low) to green (high)."""
        lo, hi = _CONF_RANGE
        norm = np.clip((conf - lo) / (hi - lo), 0.0, 1.0)
        colors = np.zeros((len(norm), 4), dtype=np.uint8)
        colors[:, 0] = ((1.0 - norm) * 255).astype(np.uint8)
        colors[:, 1] = (norm * 255).astype(np.uint8)
        colors[:, 3] = 255
        return colors

    def map_colors(self, source: np.ndarray) -> np.ndarray:
        """Per-point RGB naming which map each point came from.

        One hue per merged map, golden-angle apart so any number of maps stays distinct.
        """
        import colorsys
        palette = np.array([[int(255 * v) for v in
                             colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, 0.9, 1.0)]
                            for i in range(max(len(self.maps), 1))], dtype=np.uint8)
        return palette[source]

    @staticmethod
    def _export_cloud(path: str, points: np.ndarray, colors: np.ndarray, name: str):
        """Write one coloured point cloud as a Y-up GLB."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.PointCloud(vertices=points @ _ENU_TO_YUP.T, colors=colors),
                           geom_name=name)
        scene.export(path)

    def to_geojson(self) -> dict:
        """site.geojson: where the site's local (0, 0, 0) sits on Earth."""
        return {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [self.origin.longitude, self.origin.latitude,
                                         self.origin.altitude or 0.0]},
            "properties": {"layer": "site_origin",
                           "model": f"stages/{len(self.maps)}/scene.glb",
                           "description": "model local (0,0,0); the first merged map's origin",
                           "altitude_mode": "absolute",
                           "num_maps": len(self.maps),
                           "num_points": len(self.points),
                           "mean_confidence": float(self.conf.mean()),
                           "median_confidence": float(np.median(self.conf)),
                           "map_ids": [m.meta.name for m in self.maps]}}]}

    def load(self, maps: Dict[str, Map]) -> bool:
        """Restore a site an earlier run wrote; False if there is nothing on disk."""
        meta = os.path.join(self.path, "site.geojson")
        if not os.path.isfile(meta):
            return False
        with open(meta) as f:
            feature = json.load(f)["features"][0]
        map_ids = feature["properties"].get("map_ids", [])
        if not os.path.isfile(os.path.join(self._stage_dir(len(map_ids)), "site.npz")):
            return False

        lon, lat, alt = feature["geometry"]["coordinates"]
        self.origin = GPS(lat, lon, alt)
        self.maps = [maps[i] for i in map_ids if i in maps]
        arrays = np.load(os.path.join(self._stage_dir(len(map_ids)), "site.npz"))
        for name in _FIELDS:
            setattr(self, name, np.array(arrays[name]))
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
        self.segments = [SegmentedTarget(str(query), color, [])
                         for query, color in zip(arrays["queries"], arrays["colors"])]
        chunks = np.split(arrays["points"], np.cumsum(arrays["counts"])[:-1])
        for owner, support, score, points in zip(arrays["owner"], arrays["support"],
                                                 arrays["score"], chunks):
            self.segments[owner].instances.append(
                SegmentedInstance(points, int(support), float(score)))
