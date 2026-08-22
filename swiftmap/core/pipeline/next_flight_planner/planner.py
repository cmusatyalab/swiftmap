# Copyright (C) 2024 Carnegie Mellon University
"""
Next Flight Navigation (NFN) — confidence-difference flight planner.

Idea: a VGGT reconstruction comes with a per-point confidence. Points that pass a
*loose* confidence cut (e.g. the 60th percentile) but fail a *strict* one (e.g. the
80th percentile) are the "marginally mapped" regions -- good enough to show up, but not
yet reliable. Those are exactly the places worth re-flying.

Pipeline:
  0. Push the reconstruction through the Map's Georeference into ENU metres.
  1. Keep the points whose confidence falls in the [P_low, P_high) band.
  2. Tile the band points into fixed-footprint cells on the East/North ground plane. Each
     well-populated cell is one local cluster -- so a large weak area becomes several
     actionable viewpoints instead of one giant blob.
  3. Fit a surface normal per cluster (PCA), oriented upward.
  4. Place a suggested viewpoint at a fixed stand-off along a tilted view direction,
     looking back at the cluster center.

The planner works in ENU metres throughout, so every distance is a real metre and the
per-cluster golden-angle heading is a true compass azimuth. It requires a GPS-aligned Map.
"""

import numpy as np

from typing import Any, Dict, List, Optional
from swiftmap.core.pipeline.next_flight_planner import postprocess
from swiftmap.database.map import Map
from swiftmap.database.types import CameraPose, Cluster, Viewpoint


class NextFlightPlanner:
    def __init__(self):
        """
        params:
            low_percentile:  loose confidence cut (the "60%" map).
            high_percentile: strict confidence cut (the "80" map).
            cell_size_m:     side length of each planning cell in metres (the ground
                             footprint a single suggested viewpoint covers).
            standoff_m:      how far above the surface to place a viewpoint, in metres.
            tilt_deg:        oblique angle of suggested viewpoints away from vertical
                             (0 = straight-down nadir). Each cluster gets a different
                             azimuth so the plan covers varied viewing angles.
            min_cluster_points: discard cells with fewer band points than this.
            max_viewpoints:  cap on the number of suggested viewpoints (weakest cells first).
        """

        self.default_params = {
            "low_percentile": 60.0,
            "high_percentile": 80.0,
            "cell_size_m": 5.0,
            "standoff_m": 15.0,
            "tilt_deg": 35.0,
            "min_cluster_points": 30,
            "max_viewpoints": 20,
        }

    def run(self, map: Map, processing_params: Optional[Dict[str, Any]] = None) -> Dict:
        # process params
        params = self.default_params.copy()
        if processing_params:
            params.update(processing_params)

        # get the world points
        pt = map.get_pointcloud()
        if pt is None or pt.world_points is None:
            return self._empty("No reconstruction to plan from", params)
        georef = map.get_georeference()
        if georef is None:
            return self._empty("Map is not GPS-aligned; run align_gps() first", params)

        wp, conf = pt.flatten_points(), pt.flatten_conf()
        valid = np.isfinite(wp).all(1) & np.isfinite(conf)
        wp, conf = wp[valid], conf[valid]
        if len(wp) == 0:
            return self._empty("No valid points", params)

        # Everything below is ENU metres about the georeference origin.
        wp = georef.to_enu(wp)
        extent = float(np.linalg.norm(wp.max(0) - wp.min(0)))

        # Confidence band between the loose and strict thresholds
        p_low = float(np.percentile(conf, params["low_percentile"]))
        p_high = float(np.percentile(conf, params["high_percentile"]))
        if p_low >= p_high:
            # Confidence is near-uniform, so there is no weak band to target.
            return self._empty(
                "VGGT confidence is saturated (near-uniform across all points), so there "
                "are no low-confidence regions to target. The reconstruction is weakly "
                "constrained — capture more overlapping views and try again.",
                params, np.empty((0, 3)), p_low, p_high, extent)
        band = (conf >= p_low) & (conf < p_high)
        enhance_pts = wp[band]
        if len(enhance_pts) < params["min_cluster_points"]:
            return self._empty("Not enough band points to plan", params,
                               enhance_pts, p_low, p_high, extent)

        scene_centroid = wp.mean(0)
        east, north, up = np.eye(3)
        cell = max(float(params["cell_size_m"]), 1e-9)
        inverse = self._tile_cells(enhance_pts, scene_centroid, east, north, cell)

        standoff = float(params["standoff_m"])
        tilt = np.deg2rad(params["tilt_deg"])
        golden = np.pi * (3.0 - np.sqrt(5.0))
        clusters, viewpoints = [], []
        for gi in range(inverse.max() + 1):
            cpts = enhance_pts[inverse == gi]
            if len(cpts) < params["min_cluster_points"]:
                continue

            centroid = cpts.mean(0)
            normal = self._principal_normal(cpts - centroid)
            if np.dot(normal, up) < 0:
                normal = -normal
            radius = float(np.linalg.norm(cpts - centroid, axis=1).max())

            # Tilt off vertical per cluster so the views vary instead of all being nadir.
            k = len(clusters)
            azimuth = (k * golden) % (2.0 * np.pi)
            horiz = np.sin(azimuth) * east + np.cos(azimuth) * north
            view_dir = np.cos(tilt) * up + np.sin(tilt) * horiz
            view_dir /= (np.linalg.norm(view_dir) + 1e-9)
            position = centroid + view_dir * standoff
            rotation = self._look_at(position, centroid, up)

            clusters.append(Cluster(k, centroid, normal, int(len(cpts)), radius))
            viewpoints.append(Viewpoint(k, CameraPose(position, rotation), centroid,
                                        float(np.rad2deg(azimuth)), float(len(cpts))))

        order = np.argsort([-v.score for v in viewpoints])[:params["max_viewpoints"]]
        viewpoints = [viewpoints[i] for i in order]

        plan = {
            "viewpoints": viewpoints,
            "clusters": clusters,
            "thresholds": {
                "low_percentile": params["low_percentile"],
                "high_percentile": params["high_percentile"],
                "p_low": p_low, "p_high": p_high,
            },
            "statistics": {
                "num_enhance_points": int(len(enhance_pts)),
                "num_clusters": len(clusters),
                "scene_extent_m": extent, "cell_size_m": cell, "standoff_m": standoff,
            },
        }
        postprocess.generate_flight_plan(map, plan)
        map.write2disk()
        return plan

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tile_cells(pts: np.ndarray, origin: np.ndarray, east, north, cell: float):
        """Per-point labels from a fixed-metre grid on the East/North ground plane."""
        rel = pts - origin
        grid = np.stack([np.floor((rel @ east) / cell),
                         np.floor((rel @ north) / cell)], axis=1).astype(int)
        _, labels = np.unique(grid, axis=0, return_inverse=True)
        return labels

    @staticmethod
    def _principal_normal(centered: np.ndarray) -> np.ndarray:
        """Surface normal = direction of least variance (smallest PCA eigenvector)."""
        if len(centered) < 3:
            return np.array([0.0, 0.0, 1.0])
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        return eigvecs[:, 0]

    @staticmethod
    def _look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Camera-to-world rotation (OpenCV axes: +Z forward) looking from position at target."""
        z = target - position
        z = z / (np.linalg.norm(z) + 1e-9)
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z); x = x / (np.linalg.norm(x) + 1e-9)
        y = np.cross(z, x)
        return np.column_stack([x, y, z]).astype(np.float32)

    def _empty(self, msg, params=None, enhance_pts=None, p_low=0.0, p_high=0.0, extent=0.0) -> Dict:
        """Empty plan carrying the reason it could not be built."""
        print(f"[NFN] {msg}")
        params = params or self.default_params
        return {
            "error": msg,
            "enhance_points": enhance_pts if enhance_pts is not None else np.empty((0, 3)),
            "clusters": [], "viewpoints": [], "num_viewpoints": 0,
            "thresholds": {
                "low_percentile": params["low_percentile"],
                "high_percentile": params["high_percentile"],
                "p_low": p_low, "p_high": p_high,
            },
            "statistics": {"message": msg, "scene_extent_m": extent},
        }