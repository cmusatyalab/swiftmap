# Copyright (C) 2024 Carnegie Mellon University
"""Align a local (unitless) reconstruction to GPS.

``GpsTransformer`` fits a similarity transform (closed-form Umeyama) from the camera
trajectory to a GPS trace and builds the ``Georeference`` (in
``swiftmap.database``) that applies it. Each camera is paired with its own keyframe's
GPS fix, so correspondences are known and no nearest-neighbour matching is needed."""

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


class GpsTransformer:
    """Fits a local->GPS similarity transform (Umeyama) and builds a Georeference."""

    def run(self, map: Map, processing_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fit the local->ENU transform from the Map's keyframe GPS and attach it."""
        align_start = time.time()
        params = {"conf_threshold": constants.DEFAULT_CONF_THRESHOLD}
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
        # NOTE: altitudes are taken as-is, so they must be absolute (above sea level), never
        # relative to takeoff. A takeoff-relative zero shifts with the ground elevation of
        # each launch spot: one map stays self-consistent, but maps launched from different
        # spots then land at different heights and do not fuse in the site. TODO: revisit
        # (e.g. warn on implausibly low altitudes, or require the client to declare a datum).
        lla = np.array([[g.latitude, g.longitude, g.altitude or 0.0] for _, g in pairs], dtype=float)
        try:
            georef = self._calibrate(slam, lla)
        except Exception as e:
            return {"error": f"GPS alignment failed: {e}"}

        map.update_georeference(georef)
        postprocess.generate_geo_scene(map, params)
        align_time = time.time() - align_start
        print(f"[GPS] alignment completed in {align_time:.2f}s")
        return {"success": True, "georeference": georef, "timing": {"alignment": align_time}}

    def _calibrate(self, slam_xyz: np.ndarray, gps_lla: np.ndarray) -> Georeference:
        """Solve the local->ENU similarity transform for paired camera centres and GPS."""
        if not PYMAP3D_AVAILABLE:
            raise ImportError("pymap3d is required for GPS alignment")
        slam = np.asarray(slam_xyz, dtype=float).reshape(-1, 3)
        gps = np.asarray(gps_lla, dtype=float).reshape(-1, 3)
        if len(slam) != len(gps) or len(slam) < 3:
            raise ValueError("Need at least 3 camera centers, each paired with a GPS fix")

        lat0, lon0, alt0 = gps[0]
        e, n, u = geodetic2enu(gps[:, 0], gps[:, 1], gps[:, 2], lat0, lon0, alt0)
        enu = np.column_stack([e, n, u])

        s, R, t = self._umeyama(slam.T, enu.T, with_scale=True)
        rmse = float(np.sqrt((((s * (R @ slam.T) + t).T - enu) ** 2).sum(1).mean()))
        print(f"GPS aligned: scale={s:.4f}, RMSE={rmse:.3f} m ({len(slam)} points)")
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
        D = np.ones(3)
        if np.linalg.det(U @ Vt) < 0:  # a reflection: flip the weakest axis instead
            D[-1] = -1
        R = U @ np.diag(D) @ Vt
        s = 1.0
        if with_scale:  # the flipped axis subtracts from the scale: s = tr(DS) / var(src)
            s = (S * D).sum() / ((src_c ** 2).sum() / src.shape[1])
        t = mu_d - s * R @ mu_s
        return s, R, t
