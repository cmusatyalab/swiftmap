# Copyright (C) 2024 Carnegie Mellon University
"""Align a local (unitless) reconstruction to GPS.

``GpsTransformer`` fits a similarity transform (Umeyama + ICP) from the camera
trajectory to a GPS trace and builds the ``Georeference`` (in
``swiftmap.core.database``) that applies it."""

import time

import numpy as np
from typing import Any, Dict, Optional

from swiftmap import constants
from swiftmap.core.pipeline.gps_transformer import postprocess
from swiftmap.database.map import Map
from swiftmap.database.types import GPS, Georeference

try:
    from pymap3d import geodetic2enu
    PYMAP3D_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYMAP3D_AVAILABLE = False

try:
    from scipy.spatial import cKDTree as _KDTree
    _USE_KD = True
except ImportError:  # pragma: no cover
    _USE_KD = False


class GpsTransformer:
    """Fits a local->GPS similarity transform (Umeyama + ICP) and builds a Georeference."""

    def run(self, map: Map, processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fit the local->ENU transform from the Map's keyframe GPS and attach it."""
        align_start = time.time()
        params = {"use_icp": False, "conf_threshold": constants.DEFAULT_CONF_THRESHOLD}
        if processing_params:
            params.update(processing_params)

        pt = map.get_pointcloud()
        if pt is None or not pt.cameras:
            return {"error": "No reconstruction with camera poses yet"}

        cams = pt.camera_centers()
        pairs = [(c, g) for c, g in zip(cams, map.get_input_gps()) if g is not None]
        if len(pairs) < 3:
            return {"error": f"Only {len(pairs)} keyframes carry GPS; need at least 3"}

        slam = np.array([c for c, _ in pairs], dtype=float)
        lla = np.array([[g.latitude, g.longitude, g.altitude or 0.0] for _, g in pairs], dtype=float)
        try:
            georef = self._calibrate(slam, lla, use_icp=params["use_icp"])
        except Exception as e:
            return {"error": f"GPS alignment failed: {e}"}

        map.update_georeference(georef)
        postprocess.generate_geo_scene(map, params)
        align_time = time.time() - align_start
        print(f"[GPS] alignment completed in {align_time:.2f}s")
        return {"success": True, "georeference": georef, "timing": {"alignment": align_time}}

    def _calibrate(self, slam_xyz: np.ndarray, gps_lla: np.ndarray,
                   use_icp: bool = True) -> Georeference:
        """Solve the local->ENU similarity transform for paired camera centres and GPS."""
        if not PYMAP3D_AVAILABLE:
            raise ImportError("pymap3d is required for GPS alignment")
        slam = np.asarray(slam_xyz, dtype=float).reshape(-1, 3)
        gps = np.asarray(gps_lla, dtype=float).reshape(-1, 3)
        if len(slam) < 3 or len(gps) < 3:
            raise ValueError("Need at least 3 camera centers and 3 GPS points to align")

        lat0, lon0, alt0 = gps[0]
        e, n, u = geodetic2enu(gps[:, 0], gps[:, 1], gps[:, 2], lat0, lon0, alt0)
        enu = np.column_stack([e, n, u])

        k = min(len(slam), len(enu))
        s_idx, d_idx = self._resample_indices(k, len(slam)), self._resample_indices(k, len(enu))
        s, R, t = self._umeyama(slam[s_idx].T, enu[d_idx].T, with_scale=True)
        rmse = float(np.sqrt((((s * (R @ slam[s_idx].T) + t).T - enu[d_idx]) ** 2).sum(1).mean()))
        if use_icp:
            s, R, t, rmse = self._icp_sim3(slam, enu, s, R, t)

        print(f"GPS aligned: scale={s:.4f}, RMSE={rmse:.3f} m ({k} points)")
        return Georeference(s, R, np.asarray(t).flatten(), GPS(lat0, lon0, alt0))

    # --------------------------------------------------------- registration math
    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
        """Closed-form similarity (Sim3) alignment of (3, N) corresponding points.
        Returns (s, R, t) minimizing ||dst - (s*R*src + t)||."""
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

    @staticmethod
    def _nearest(src: np.ndarray, dst: np.ndarray):
        """For each point in src (N,3), index of nearest in dst (M,3)."""
        if _USE_KD:
            return _KDTree(dst).query(src)[1]
        diff = src[:, None, :] - dst[None, :, :]
        return (diff ** 2).sum(-1).argmin(1)

    @classmethod
    def _icp_sim3(cls, src: np.ndarray, dst: np.ndarray, s0, R0, t0,
                  max_iter: int = 30, eps: float = 1e-6):
        """Refine a Sim(3) fit by alternating nearest-neighbour + Umeyama.

        src (N,3) points, dst (M,3) target cloud. Returns (s, R, t, rmse)."""
        s, R, t = s0, R0, t0
        rmse = float("inf")
        prev = None
        for _ in range(max_iter):
            src_t = (s * (R @ src.T) + t).T
            idx = cls._nearest(src_t, dst)
            rmse = float(np.sqrt(((src_t - dst[idx]) ** 2).sum(1).mean()))
            s, R, t = cls._umeyama(src.T, dst[idx].T, with_scale=True)
            if prev is not None and abs(prev - rmse) < eps:
                break
            prev = rmse
        return s, R, t, rmse

    @staticmethod
    def _resample_indices(k: int, n: int) -> np.ndarray:
        """k evenly-spaced indices into a length-n sequence."""
        return np.linspace(0, n - 1, k).round().astype(int)
