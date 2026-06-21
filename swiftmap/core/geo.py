# Copyright (C) 2024 Carnegie Mellon University
"""
Geo-alignment: align a VGGT (local, unitless) reconstruction to GPS.

A VGGT reconstruction has no absolute scale, orientation, or position. Given the
per-keyframe camera centers (local xyz, ordered along the flight) and a GPS
trajectory (lat/lon/alt, ordered — possibly a different count and not synced to the
frames), we solve a similarity transform (scale s, rotation R, translation t) that
maps local coordinates into a local ENU frame:

    ENU = s * R * xyz + t

then convert ENU -> geodetic to get real lat/lon/alt for any local point (NFN
viewpoints, the point cloud, camera poses).

The solve is trajectory-to-trajectory (no per-frame correspondence needed):
Umeyama (closed-form SVD) for an initial fit, then ICP (nearest-neighbour against
the GPS cloud) to absorb the count/timing mismatch. Ported from TerraSLAM's
SLAM-GPS alignment.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple

try:
    from pymap3d import geodetic2enu, enu2geodetic
    PYMAP3D_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYMAP3D_AVAILABLE = False

try:
    from scipy.spatial import cKDTree as _KDTree
    _USE_KD = True
except ImportError:  # pragma: no cover
    _USE_KD = False


# --------------------------------------------------------------------- Umeyama
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Closed-form similarity alignment. src, dst are (3, N) corresponding points.

    Returns (s, R, t) minimizing ||dst - (s*R*src + t)||.
    """
    mu_s = src.mean(axis=1, keepdims=True)
    mu_d = dst.mean(axis=1, keepdims=True)
    src_c, dst_c = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(dst_c @ src_c.T / src.shape[1])
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflection fix
        Vt[-1] *= -1
        R = U @ Vt
    s = 1.0
    if with_scale:
        s = S.sum() / ((src_c ** 2).sum() / src.shape[1])
    t = mu_d - s * R @ mu_s
    return s, R, t


# ------------------------------------------------------------------------- ICP
def _nearest(src: np.ndarray, dst: np.ndarray):
    """For each point in src (N,3), index of nearest in dst (M,3)."""
    if _USE_KD:
        return _KDTree(dst).query(src)[1]
    diff = src[:, None, :] - dst[None, :, :]
    return (diff ** 2).sum(-1).argmin(1)


def icp_sim3(src: np.ndarray, dst: np.ndarray, s0, R0, t0,
             max_iter: int = 30, eps: float = 1e-6):
    """Refine a Sim(3) fit by alternating nearest-neighbour + Umeyama.

    src (N,3) local points, dst (M,3) target ENU cloud. Returns (s, R, t, rmse).
    """
    s, R, t = s0, R0, t0
    rmse = float("inf")
    prev = None
    for _ in range(max_iter):
        src_t = (s * (R @ src.T) + t).T
        idx = _nearest(src_t, dst)
        rmse = float(np.sqrt(((src_t - dst[idx]) ** 2).sum(1).mean()))
        s, R, t = umeyama(src.T, dst[idx].T, with_scale=True)
        if prev is not None and abs(prev - rmse) < eps:
            break
        prev = rmse
    return s, R, t, rmse


def _resample(k: int, n: int) -> np.ndarray:
    """k evenly-spaced indices into a length-n sequence."""
    return np.linspace(0, n - 1, k).round().astype(int)


def calibrate(slam_xyz: np.ndarray, gps_lla: np.ndarray,
              use_icp: bool = True) -> Dict[str, Any]:
    """Solve the local->GPS similarity transform.

    Args:
        slam_xyz: (N, 3) local camera centers, in capture order.
        gps_lla:  (M, 3) GPS as [lat, lon, alt], in capture order. N and M may differ.
        use_icp:  refine the Umeyama fit with ICP (recommended when N != M / unsynced).

    Returns a config dict: {scale, rotation, translation, lat0, lon0, alt0,
    init_rmse, rmse, num_points} (RMSE in meters).
    """
    if not PYMAP3D_AVAILABLE:
        raise ImportError("pymap3d is required for GPS alignment")

    slam = np.asarray(slam_xyz, dtype=float).reshape(-1, 3)
    gps = np.asarray(gps_lla, dtype=float).reshape(-1, 3)
    if len(slam) < 3 or len(gps) < 3:
        raise ValueError("Need at least 3 camera centers and 3 GPS points to align")

    lat0, lon0, alt0 = gps[0]
    e, n, u = geodetic2enu(gps[:, 0], gps[:, 1], gps[:, 2], lat0, lon0, alt0)
    enu = np.column_stack([e, n, u])

    # Initial fit on an equal-count resampling of both trajectories.
    k = min(len(slam), len(enu))
    s_idx, d_idx = _resample(k, len(slam)), _resample(k, len(enu))
    s0, R0, t0 = umeyama(slam[s_idx].T, enu[d_idx].T, with_scale=True)
    init_fit = (s0 * (R0 @ slam[s_idx].T) + t0).T
    init_rmse = float(np.sqrt(((init_fit - enu[d_idx]) ** 2).sum(1).mean()))

    if use_icp:
        s, R, t, rmse = icp_sim3(slam, enu, s0, R0, t0)
    else:
        s, R, t, rmse = s0, R0, t0, init_rmse

    return {
        "scale": float(s),
        "rotation": np.asarray(R).tolist(),
        "translation": np.asarray(t).flatten().tolist(),
        "lat0": float(lat0), "lon0": float(lon0), "alt0": float(alt0),
        "init_rmse": init_rmse, "rmse": float(rmse),
        "num_points": int(k),
    }


# ------------------------------------------------------------------- applier
class GpsTransform:
    """Applies a calibrated similarity transform: local xyz -> [lat, lon, alt]."""

    def __init__(self, cfg: Dict[str, Any]):
        if not PYMAP3D_AVAILABLE:
            raise ImportError("pymap3d is required for GPS transformation")
        self.cfg = cfg
        self.s = float(cfg["scale"])
        self.R = np.asarray(cfg["rotation"], dtype=float)
        self.t = np.asarray(cfg["translation"], dtype=float).reshape(3, 1)
        self.lat0, self.lon0, self.alt0 = cfg["lat0"], cfg["lon0"], cfg["alt0"]

    def to_lla(self, xyz: np.ndarray) -> np.ndarray:
        """Local xyz -> GPS. Accepts (3,) or (N, 3); returns the same shape with
        [lat, lon, alt]."""
        xyz = np.asarray(xyz, dtype=float)
        single = xyz.ndim == 1
        pts = xyz.reshape(1, 3) if single else xyz
        enu = (self.s * self.R @ pts.T + self.t).T
        lat, lon, alt = enu2geodetic(enu[:, 0], enu[:, 1], enu[:, 2],
                                     self.lat0, self.lon0, self.alt0)
        out = np.column_stack([lat, lon, alt])
        return out[0] if single else out

    @classmethod
    def from_calibration(cls, slam_xyz: np.ndarray, gps_lla: np.ndarray,
                         use_icp: bool = True) -> Tuple["GpsTransform", Dict[str, Any]]:
        cfg = calibrate(slam_xyz, gps_lla, use_icp=use_icp)
        return cls(cfg), cfg


def load_gps_csv(path: str) -> np.ndarray:
    """Load a GPS CSV with columns latitude, longitude, altitude (or height) into
    an (M, 3) array of [lat, lon, alt]."""
    import csv
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        lat_c = cols.get("latitude") or cols.get("lat")
        lon_c = cols.get("longitude") or cols.get("lon")
        alt_c = cols.get("altitude") or cols.get("height") or cols.get("alt")
        if not (lat_c and lon_c and alt_c):
            raise ValueError(f"GPS CSV must have latitude/longitude/altitude columns; got {reader.fieldnames}")
        for r in reader:
            rows.append([float(r[lat_c]), float(r[lon_c]), float(r[alt_c])])
    return np.asarray(rows, dtype=float)
