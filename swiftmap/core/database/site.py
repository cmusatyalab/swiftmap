# Copyright (C) 2024 Carnegie Mellon University

"""The ``Site``: a database's one growing map."""

import glob
import os
from datetime import datetime

from swiftmap.core.database.map import Map
import numpy as np
import pymap3d

from swiftmap.core.database.georeference import Georeference


class Site(Map):
    """A ``Map`` that grows: each stored map is merged into it, in place."""

    def __init__(self, path: str, name: str = "map"):
        super().__init__(path)
        self.name = name

    def __repr__(self):
        return f"Site({self.name!r}, {len(self.sources)} map(s))"

    @property
    def sources(self) -> list:
        """Tags of the maps merged in so far."""
        return list(self.metadata.source_maps)

    def grow(self, new_map: Map, conf_thres: float = 50.0, voxel_size: float = 0.1,
             created: datetime = None) -> "Site":
        """Merge ``new_map`` into the site (origin pinned to the site) and rewrite its data.

        Views are rendered on demand through the session."""
        parts = [self.load()] if self.exists() else []
        parts.append(new_map.load(conf_thres))
        before = sum(len(p[0]) for p in parts)
        pts, cols, conf, gt, frames = _merge(parts, voxel_size=voxel_size)
        print(f"[site] grow with '{new_map.tag}': {before:,} -> {len(pts):,} pts "
              f"(voxel {voxel_size:g} m, conf>={conf_thres:g}p, "
              f"origin {gt.lat0:.6f},{gt.lon0:.6f},{gt.alt0:.1f})")
        for stale in glob.glob(os.path.join(self.path, "*_view_c*.glb")):
            os.remove(stale)
        return self.write(pts, cols, conf, gt, frames,
                          source_maps=self.sources + [new_map.tag],
                          site=self.name, created=created)


def voxel_merge(points, colors, conf, voxel_size: float):
    """Collapse points sharing a ``voxel_size`` grid cell into one, confidence-weighted.

    Position and color become the confidence-weighted mean of the cell; the merged
    confidence is the cell's max. ``voxel_size <= 0`` returns the input unchanged.
    """
    points = np.asarray(points, dtype=float)
    if voxel_size <= 0 or len(points) == 0:
        return points, np.asarray(colors), np.asarray(conf, dtype=float)

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
    return pos, np.clip(col, 0, 255).astype(np.uint8), mconf


# ---------------------------------------------------------------------- meshing


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
    """Camera frames into the common ENU frame about ``origin_lla``."""
    if not frames:
        return []
    r_frame = _enu_frame_rotation(gt, origin_lla) @ gt.R
    out = []
    for fr in frames:
        c2 = gt.to_enu(np.asarray(fr["camera_position_world"], float), origin_lla)[0]
        r2 = np.asarray(fr["rotation_matrix"], float) @ r_frame.T
        t2 = -r2 @ c2
        out.append({**fr, "camera_position_world": c2.tolist(), "rotation_matrix": r2.tolist(),
                    "translation_vector": t2.tolist(),
                    "extrinsic_matrix": np.hstack([r2, t2.reshape(3, 1)]).tolist()})
    return out


def _merge(parts, origin=None, voxel_size: float = 0.1):
    """GPS-co-register ``(points, colors, conf, transform, frames)`` parts into one cloud
    in the common ENU frame about ``origin`` (default: the first part's), voxel-collapsed."""
    parts = list(parts)
    first = parts[0][3]
    origin = origin or (first.lat0, first.lon0, first.alt0)
    pts, cols, conf, frames = [], [], [], []
    for p, c, cf, gt, frs in parts:
        pts.append(gt.to_enu(p, origin))
        cols.append(np.asarray(c))
        conf.append(np.asarray(cf, dtype=float))
        frames += _transform_frames(frs, gt, origin)
    mpts, mcols, mconf = voxel_merge(np.vstack(pts), np.vstack(cols),
                                     np.concatenate(conf), voxel_size)
    identity = Georeference({"scale": 1.0, "rotation": np.eye(3).tolist(),
                             "translation": [0.0, 0.0, 0.0],
                             "lat0": origin[0], "lon0": origin[1], "alt0": origin[2]})
    return mpts, mcols, mconf, identity, frames
