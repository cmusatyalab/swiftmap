# Copyright (C) 2024 Carnegie Mellon University

"""``Georeference``: a map's calibrated Sim(3) -- local xyz <-> GPS."""

import numpy as np
import pymap3d


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
