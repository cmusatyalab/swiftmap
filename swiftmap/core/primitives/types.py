# Copyright (C) 2024 Carnegie Mellon University

"""Shared in-memory types. ``MapData`` (points + colors + conf + Georeference + camera
frames) is the point-cloud currency across the core; ``merge`` GPS-co-registers a list
of MapData into one ENU cloud and voxel-collapses duplicates."""

from dataclasses import dataclass, field

import numpy as np
import pymap3d

from swiftmap.core.primitives import geometry


class Georeference:
    """A map's georeference: applies a calibrated Sim(3), local xyz -> [lat, lon, alt]."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.s = float(cfg["scale"])
        self.R = np.asarray(cfg["rotation"], dtype=float)
        self.t = np.asarray(cfg["translation"], dtype=float).reshape(3, 1)
        self.lat0, self.lon0, self.alt0 = cfg["lat0"], cfg["lon0"], cfg["alt0"]

    def to_lla(self, xyz: np.ndarray) -> np.ndarray:
        """Local xyz -> [lat, lon, alt]. Accepts (3,) or (N,3), returns the same shape."""
        xyz = np.asarray(xyz, dtype=float)
        single = xyz.ndim == 1
        pts = xyz.reshape(1, 3) if single else xyz
        enu = (self.s * self.R @ pts.T + self.t).T
        lat, lon, alt = pymap3d.enu2geodetic(enu[:, 0], enu[:, 1], enu[:, 2],
                                             self.lat0, self.lon0, self.alt0)
        out = np.column_stack([lat, lon, alt])
        return out[0] if single else out

    def to_enu(self, xyz, origin_lla) -> np.ndarray:
        """Local xyz -> ENU meters about ``origin_lla`` (via LLA)."""
        lla = np.atleast_2d(self.to_lla(xyz))
        e, n, u = pymap3d.geodetic2enu(lla[:, 0], lla[:, 1], lla[:, 2], *origin_lla)
        return np.column_stack([e, n, u])


def _enu_frame_rotation(gt: Georeference, origin_lla) -> np.ndarray:
    """Rotation from a map's own ENU frame into the common ENU frame (tangent tilt)."""
    probe = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    lat, lon, alt = pymap3d.enu2geodetic(probe[:, 0], probe[:, 1], probe[:, 2],
                                         gt.lat0, gt.lon0, gt.alt0)
    e, n, u = pymap3d.geodetic2enu(lat, lon, alt, *origin_lla)
    p = np.column_stack([e, n, u])
    u_, _, vt = np.linalg.svd((p[1:] - p[0]).T)
    return u_ @ vt


def _transform_frames(frames, gt: Georeference, origin_lla):
    """Transform camera frames into the common ENU frame about ``origin_lla``."""
    if not frames:
        return []
    r_frame = _enu_frame_rotation(gt, origin_lla) @ gt.R
    out = []
    for fr in frames:
        c2 = gt.to_enu(np.asarray(fr["camera_position_world"], float), origin_lla)[0]
        r2 = np.asarray(fr["rotation_matrix"], float) @ r_frame.T
        t2 = -r2 @ c2
        out.append({**fr,
                    "camera_position_world": c2.tolist(),
                    "rotation_matrix": r2.tolist(),
                    "translation_vector": t2.tolist(),
                    "extrinsic_matrix": np.hstack([r2, t2.reshape(3, 1)]).tolist()})
    return out


@dataclass
class MapData:
    """A point cloud with its georeference and cameras -- the merge currency."""

    points: np.ndarray                 # (N, 3)
    colors: np.ndarray                 # (N, 3) uint8
    conf: np.ndarray                   # (N,)
    transform: Georeference
    frames: list = field(default_factory=list)

    @property
    def origin(self) -> tuple:
        """The (lat0, lon0, alt0) origin of this map's transform."""
        gt = self.transform
        return (gt.lat0, gt.lon0, gt.alt0)

    def __len__(self) -> int:
        return len(self.points)

    def filtered(self, percentile) -> "MapData":
        """A copy keeping only points at/above the ``percentile`` confidence cut."""
        keep = geometry.confidence_mask(self.conf, percentile)
        return MapData(self.points[keep], self.colors[keep], self.conf[keep],
                       self.transform, self.frames)

    def to_common_enu(self, origin_lla) -> np.ndarray:
        """This map's points in ENU meters about ``origin_lla``."""
        return self.transform.to_enu(self.points, origin_lla)

    def pointcloud(self):
        """A trimesh PointCloud of this map."""
        return geometry.pointcloud(self.points, self.colors)

    @staticmethod
    def merge(maps, origin=None, voxel_size: float = 0.1) -> "MapData":
        """GPS-co-register ``maps`` into one cloud in the common ENU frame about ``origin``
        (defaults to the first map's origin), collapsing near-duplicates by ``voxel_size``.

        Returns a MapData whose transform is the identity at ``origin`` -- the merged
        cloud already lives in that ENU frame."""
        maps = list(maps)
        origin = origin or maps[0].origin
        pts, cols, conf, frames = [], [], [], []
        for m in maps:
            pts.append(m.to_common_enu(origin))
            cols.append(np.asarray(m.colors))
            conf.append(np.asarray(m.conf, dtype=float))
            frames += _transform_frames(m.frames, m.transform, origin)
        mpts, mcols, mconf = geometry.voxel_merge(np.vstack(pts), np.vstack(cols),
                                                  np.concatenate(conf), voxel_size)
        identity = Georeference({"scale": 1.0, "rotation": np.eye(3).tolist(),
                                 "translation": [0.0, 0.0, 0.0],
                                 "lat0": origin[0], "lon0": origin[1], "alt0": origin[2]})
        return MapData(mpts, mcols, mconf, identity, frames)


class Reconstruction(dict):
    """A backbone's prediction dict with typed accessors (schema in mapper/postprocess).

    A plain ``dict`` subclass, so it is a drop-in wherever predictions flowed as a
    dict; the properties give typed access and ``to_mapdata`` builds the point cloud.
    """

    @classmethod
    def wrap(cls, obj) -> "Reconstruction":
        return obj if isinstance(obj, cls) else cls(obj)

    @property
    def world_points(self) -> np.ndarray:
        return np.asarray(self["world_points"])

    @property
    def world_points_conf(self) -> np.ndarray:
        return np.asarray(self["world_points_conf"])

    @property
    def images(self) -> np.ndarray:
        return np.asarray(self["images"])

    @property
    def camera_positions(self):
        return np.asarray(self["camera_positions"]) if "camera_positions" in self else None

    @property
    def has_points(self) -> bool:
        return "world_points" in self and "world_points_conf" in self

    def to_mapdata(self, transform, frames=None, conf_percentile=None) -> "MapData":
        """Build a confidence-filtered MapData from this reconstruction."""
        pts = self.world_points.reshape(-1, 3)
        cols = geometry.flatten_colors(self.images)
        keep = np.isfinite(pts).all(1)
        conf = self.world_points_conf.reshape(-1) if self.has_points else np.ones(len(pts))
        if conf_percentile:
            keep &= geometry.confidence_mask(conf, conf_percentile)
        return MapData(pts[keep], cols[keep], conf[keep], transform, frames or [])

